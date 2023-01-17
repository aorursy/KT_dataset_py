!apt-get install p7zip
!p7zip -d -f -k /kaggle/input/mercari-price-suggestion-challenge/train.tsv.7z
!p7zip -d -f -k /kaggle/input/mercari-price-suggestion-challenge/test.tsv.7z
!unzip ../input/mercari-price-suggestion-challenge/test_stg2.tsv.zip
import os,sys
import pandas as pd
import numpy as np
root_dir = os.getcwd()
model_dir = os.path.join(root_dir,'model')
# read in unpacked training data
df_train = pd.read_csv('train.tsv', sep='\t')
df_train.shape
df_train['name']=df_train['name'].astype(str)
df_train['category_name']=df_train['category_name'].astype(str)
df_train['item_description']=df_train['item_description'].astype(str)
df_train['category_text'] = df_train['category_name'].apply(lambda x: ' '.join(str(x).split('/')))
df_train['text'] = df_train[['name','category_text','item_description']].apply(lambda x: ''.join(x), axis=1)
# convert to lowercase
df_train['text'] = df_train['text'] .apply(lambda x: x.lower())
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load('../input/bpe-model-text-lower-10k/bpe_model_text_lower_10k.model')
offset = 1
df_train['tokenized'] = df_train['text'].apply(lambda x: [k+offset for k in sp.EncodeAsIds(x)])
df_train['token_cnt'] = df_train['tokenized'].apply(len)
df_train['token_cnt'].quantile(.90)
import os,sys
import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Activation
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
token_truncate_len = 90
# Create an encoded, truncated array
train_np = pad_sequences(df_train['tokenized'].values,maxlen=token_truncate_len,truncating='post')
train_np.shape
X_train = train_np
y_train = df_train['price'].values
max_features = 10000+1
# arbitrary choice, needs to be refined 
embedding_dimension = 20
from keras.optimizers import Adam
from keras.losses import mean_squared_logarithmic_error
from keras.layers import Bidirectional
model=Sequential()
model.add(
    Embedding(
        max_features,
        embedding_dimension,
        input_length=token_truncate_len
    )
)
#model.add(Bidirectional(CuDNNLSTM(embedding_dimension)))
model.add(Bidirectional(LSTM(embedding_dimension)))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('linear'))

opt = Adam(lr=1e-3, decay=1e-3 / 200)

optimizer = opt
model.compile(
    loss = mean_squared_logarithmic_error,
    optimizer=opt)

%%time
model.fit(
    X_train,
    y_train,
    epochs=1
)
# export da model
model.save('mercari_bilstm_model.h5')
z_train = model.predict(X_train)
z_train = z_train.reshape(-1,)
def rmsle(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.sqrt(np.mean(np.power(np.log1p(y_true + 1) - np.log1p(y_pred + 1), 2)))
rmsle(y_train,z_train)
df_test = pd.read_csv('test_stg2.tsv', sep='\t')
df_test['name']=df_test['name'].astype(str)
df_test['category_name']=df_test['category_name'].astype(str)
df_test['item_description']=df_test['item_description'].astype(str)
df_test['category_text'] = df_test['category_name'].apply(lambda x: ' '.join(str(x).split('/')))
df_test['text'] = df_test[['name','category_text','item_description']].apply(lambda x: ''.join(x), axis=1)
# convert to lowercase
df_test['text'] = df_test['text'] .apply(lambda x: x.lower())
offset = 1
df_test['tokenized'] = df_test['text'].apply(lambda x: [k+offset for k in sp.EncodeAsIds(x)])
X_test = pad_sequences(df_test['tokenized'].values,maxlen=token_truncate_len,truncating='post')
X_test.shape
z_test = model.predict(X_test)
z_test = z_test.reshape(-1,)
test_result_df = pd.DataFrame(data={'test_id':df_test.index,'price':z_test})
# create submission
test_result_df.to_csv("submission.csv", index = False)

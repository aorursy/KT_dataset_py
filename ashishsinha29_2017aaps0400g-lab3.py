# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import os
import sys
import numpy as np
import pandas as pd
import keras.backend as K
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Model,Sequential
from keras.layers import Dense, Embedding, Input,BatchNormalization,CuDNNGRU,MaxPool1D,Flatten
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout,Conv1D,GRU,TimeDistributed
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score

train=pd.read_csv('/kaggle/input/nnfl-lab-3-nlp/nlp_train.csv')
test=pd.read_csv('/kaggle/input/nnfl-lab-3-nlp/_nlp_test.csv')
test_original=pd.read_csv("/kaggle/input/nnfl-lab-3-nlp/_nlp_test.csv")
import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    text=re.sub(r"\b\d+\b", "", text)
    return text
train['tweet']=train['tweet'].apply(preprocessor)
test['tweet']=test['tweet'].apply(preprocessor)
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer 
stop=stopwords.words('english')
lemmatizer = WordNetLemmatizer() 
porter=PorterStemmer()

def tokenizer_porter(text):
    return [lemmatizer.lemmatize(word) for word in text.split()]
def remove_stopwords(text):
    return [w for w in text if w not in stop]
def combine(text):
    return  ' '.join(text)
train['tweet']=train['tweet'].apply(tokenizer_porter).apply(remove_stopwords).apply(combine)
test['tweet']=test['tweet'].apply(tokenizer_porter).apply(remove_stopwords).apply(combine)
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 10000
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 15

word2vec={}
with open(os.path.join('/kaggle/input/glove-global-vectors-for-word-representation/glove.twitter.27B.%sd.txt' % EMBEDDING_DIM),errors='ignore',encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec
    f.close()
print ('found %s word vectors.' %len(word2vec))     
print ('found %s word vectors.' %len(word2vec))     
sent = train['tweet'].values
sent = np.append(sent, test['tweet'].values)
sentences = train['tweet'].values
targets=train['offensive_language'].values
targets
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sent)
sequences = tokenizer.texts_to_sequences(sentences)
word2idx = tokenizer.word_index
print('Found %s unique tokens.' %len(word2idx))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
data.shape
test_data=pad_sequences(tokenizer.texts_to_sequences(test['tweet']),maxlen=MAX_SEQUENCE_LENGTH)
test_data.shape
num_words = min(MAX_VOCAB_SIZE,len(word2idx)+1)
embedding_matrix = np.zeros((num_words,EMBEDDING_DIM))
for word,i in word2idx.items():
    if i<MAX_VOCAB_SIZE:
        embedding_vector= word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
y=to_categorical(targets)
embedding_layer=Embedding(
num_words,
EMBEDDING_DIM,
weights=[embedding_matrix],
input_length=MAX_SEQUENCE_LENGTH,
trainable=True)
def custom_activation(x):
    return (K.sigmoid(x) * 3)
input_=Input(shape=(MAX_SEQUENCE_LENGTH,))
model = Sequential()
model.add(Embedding(
num_words,
EMBEDDING_DIM,
weights=[embedding_matrix],
input_length=MAX_SEQUENCE_LENGTH,
trainable=False))
model.add(Bidirectional(LSTM(256, return_sequences=True,name='lstm_layer',dropout=0.25,recurrent_dropout=0.1)))
model.add(GlobalMaxPool1D())
model.add(Dense(1,activation=custom_activation))

weights={
0:(1-len(targets[targets==0])/len(targets))/3,
1:(1-len(targets[targets==1])/len(targets))/3,
2:(1-len(targets[targets==2])/len(targets))/3,
3:(1-len(targets[targets==3])/len(targets))/3}
weights
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
model.compile(
loss=root_mean_squared_error,
optimizer=Adam(lr=0.01),
metrics=['accuracy','mse']),
model.summary()
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


file_path = "best_model.h5"
ckpt = ModelCheckpoint(file_path, monitor='val_loss', verbose=1,
                        save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=5)

callbacks = [early, learning_rate_reduction,ckpt]
model.fit(
data,
targets,
class_weight=weights,
batch_size=BATCH_SIZE,
epochs=5,
callbacks=callbacks,
validation_split=0.1)
preds = model.predict(test_data)
for i in range(len(test_original)):
    test_original.iloc[i,1]=preds[i][0]
df = test_original
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(df)

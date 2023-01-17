import pandas as pd
import time
import gc
text_cols = ['param_1','param_2','param_3','title','description']
print('loading train...')
train = pd.read_csv('../input/train.csv', index_col = 'item_id', usecols = text_cols + ['item_id','image_top_1'])
train_indices = train.index
print('loading test')
test = pd.read_csv('../input/test.csv', index_col = 'item_id', usecols = text_cols + ['item_id','image_top_1'])
test_indices = test.index
print('concat dfs')
df = pd.concat([train,test])
nan_indices = df[pd.isnull(df['image_top_1'])].index
not_nan_indices = df[pd.notnull(df['image_top_1'])].index

#df = df[pd.notnull(df['image_top_1'])]

del train, test
print('cleaning text')

for col in text_cols:
    df[col] = df[col].fillna('nan').astype(str)
print('concat text')
df['text'] = df[text_cols].apply(lambda x: ' '.join(x), axis=1)
df.drop(text_cols,axis = 1, inplace = True)
from keras.preprocessing import text
from keras.preprocessing.sequence import pad_sequences

max_features = 100000 # max amount of words considered
max_len = 100 #maximum length of text
dim = 100 #dimension of embedding


print('tokenizing...',end='')
tic = time.time()
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(df['text'].values))
toc = time.time()
print('done. {}'.format(toc-tic))

col = 'text'
print("   Transforming {} to seq...".format(col))
tic = time.time()
df[col] = tokenizer.texts_to_sequences(df[col])
toc = time.time()
print('done. {}'.format(toc-tic))

print('padding X_train')
tic = time.time()
X_train = pad_sequences(df.loc[not_nan_indices,col], maxlen=max_len)
toc = time.time()
print('done. {}'.format(toc-tic))

print('padding X_nan')
tic = time.time()
X_nan = pad_sequences(df.loc[nan_indices,col], maxlen=max_len)
toc = time.time()
print('done. {}'.format(toc-tic))

df.drop(['text'], axis = 1, inplace=True)
y = df.loc[not_nan_indices,'image_top_1'].values

import numpy as np
from keras.layers import Input,PReLU,BatchNormalization, GlobalMaxPooling1D, GlobalAveragePooling1D, CuDNNGRU, Bidirectional, Dense, Embedding
from keras.layers import Concatenate, Flatten, Bidirectional
from keras.optimizers import Adam
from keras.initializers import he_uniform
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping


from keras.losses import categorical_crossentropy, sparse_categorical_crossentropy



def all_pool(tensor):
    avg_tensor = GlobalAveragePooling1D()(tensor)
    max_tensor = GlobalMaxPooling1D()(tensor)
    res_tensor = Concatenate()([avg_tensor, max_tensor])
    return res_tensor

def build_model():
    inp = Input(shape=(max_len,))

    embedding = Embedding(max_features + 1, dim)(inp)
    x = Bidirectional(CuDNNGRU(64,return_sequences=True))(embedding)
    x = all_pool(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation = 'relu')(x)
    out = Dense(3067, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)

    model.compile(optimizer=Adam(lr=0.0005), loss=sparse_categorical_crossentropy)
    return model

model = build_model()
model.summary()

early_stop = EarlyStopping(patience=2)
check_point = ModelCheckpoint('model.hdf5', monitor = "val_loss", mode = "min", save_best_only = True, verbose = 1)

history = model.fit(X_train, y, batch_size = 512, epochs = 10,
                verbose = 1, validation_split=0.1,callbacks=[early_stop,check_point])


id2word = {tokenizer.word_index[word]:word for word in tokenizer.word_index}
weights = model.layers[1].get_weights()[0]
embedding_dict = {}
for id in id2word:
    if id <= weights.shape[0]-1:
        embedding_dict[id2word[id]] = weights[id]


import pickle
with open('embedding_dict.p','wb') as f:
    pickle.dump(embedding_dict,f)
preds = model.predict(X_nan,verbose=1)
k = 0
classes = np.zeros(shape=np.argmax(preds,axis = 1).shape)
for i in range(preds.shape[0]):
    if np.max(preds[i]) > 0.1:
        k+=1
        classes[i] = np.argmax(preds[i])
    else:
        classes[i] = np.nan
df.loc[nan_indices,'image_top_1'] = classes

df.loc[train_indices].to_csv('train_image_top_1_features.csv')
df.loc[test_indices].to_csv('test_image_top_1_features.csv')

import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder

import re

from keras.models import Model

from keras.layers import Dense, Flatten, Embedding

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM

import gc

import keras

from keras.layers import Dense, Concatenate

from keras.layers import *

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

from keras import losses

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation

from keras.layers import Bidirectional, GlobalMaxPool1D

from keras import optimizers

from itertools import product

from keras.models import load_model

cust_df = pd.read_csv("/kaggle/input/parisdata/profilesniche_prepared.csv")

perf_df = pd.read_csv("/kaggle/input/parisprodaccords/products_finals_with_accords.csv")

ratings_df = pd.read_csv("/kaggle/input/kernel70b3382677/ratings_copy.csv")
cust_df.shape
cust_df.head()
cust_df = cust_df[['IDcustomer', 'text']]
cust_df.isna().sum()
cust_df['text'].fillna("unknown", inplace = True)
perf_df.shape
perf_df.head()
perf_df[['0', '1', '2', '3']]
#removeing names infromt of nodes



vals = ['Top0', 'Top1', 'Top2', 'Top3', 'Middle0', 'Middle1', 'Middle2']

for i in ['0', '1', '2', '3']:

    print(i)

    for val in vals:

        perf_df[i] = perf_df[i].apply(lambda x: x.replace(val,''))
perf_df['0'].nunique(), perf_df['1'].nunique(), perf_df['2'].nunique(), perf_df['3'].nunique(), 
perf_df[['0', '1', '2', '3']].tail()
perf_df[['url', '0', '1', '2', '3']].to_csv("perfume_nodes.csv", index=None)
dummy = list(perf_df['0']) + list(perf_df['1']) +  list(perf_df['2']) +  list(perf_df['3']) 
dummies = set(dummy)
len(dummies)
nodes_encoding = {i:j for j, i in enumerate(dummies)}
nodes_encoding['Cetalox']
perf_df['Top0'] =  perf_df['0'].apply(lambda x:nodes_encoding[x])

perf_df['Top1'] =  perf_df['1'].apply(lambda x:nodes_encoding[x])

perf_df['Top2'] =  perf_df['2'].apply(lambda x:nodes_encoding[x])

perf_df['Top3'] =  perf_df['3'].apply(lambda x:nodes_encoding[x])



del perf_df['0']

del perf_df['1']

del perf_df['2']

del perf_df['3']
perf_df.head()
del perf_df['title']
perf_df.head()
cust_df.head()
ratings_df.head()
for col in ['avatar_img', 'date', 'karma', 'name_perfume', 'rew', 'username']:

    del ratings_df[col]
ratings_df.head()
def get_user_id(x):

    vals = re.findall(r'\d+', x)

    return vals[0]



ratings_df['IDcustomer'] = ratings_df['userlink'].apply(lambda x: get_user_id(x))
del ratings_df['userlink']
ratings_df.head()
cust_df.head()
perf_df.head()
url_encoding = {}

for i, j in enumerate(perf_df['url']):

    url_encoding[j] = i
perf_df['ID_perfume'] = perf_df['url'].apply(lambda x:url_encoding[x])
perf_df.head()
ratings_df.head()
gc.collect()
def get_url_encoding(x):

    try:

        vals = url_encoding[x]        

    except:

        vals = None

    return vals
ratings_df['ID_perfume'] = ratings_df['url'].apply(lambda x: get_url_encoding(x))
ratings_df.tail()
ratings_df.isna().sum()
ratings_df = ratings_df[ratings_df['ID_perfume'].isna()== False]
ratings_df.shape
del ratings_df['url']

del perf_df['url']
cust_df.head()
perf_df.head()
ratings_df.head()
ratings_df['ID_perfume'] = ratings_df['ID_perfume'].astype('int')
cust_df.shape, perf_df.shape, ratings_df.shape
ratings_df['IDcustomer'].dtypes, cust_df['IDcustomer'].dtypes 
ratings_df['IDcustomer'] = ratings_df['IDcustomer'].astype('int')
ratings_df.head()
del ratings_df['text']
existing_df = pd.merge(ratings_df, cust_df, on='IDcustomer', how='left')
existing_df = pd.merge(existing_df, perf_df, on='ID_perfume', how='left')
existing_df.shape
existing_df.head()
existing_df.isna().sum()
existing_df['text'].fillna('unknown', inplace = True)
existing_df.head()
combination_existed = existing_df[['IDcustomer', 'ID_perfume']]
dummy = existing_df['text'].apply(lambda x :len(x.split()))
dummy.describe()
def clean_text(x):

    x = x.lower()

    x = re.sub('[^A-Za-z0-9]+', ' ', x)

    return x
existing_df['text'] = existing_df['text'].apply(lambda x:clean_text(x))
gc.collect()
## some config values 

embed_size = 300 # how big is each word vector

max_features = 100000 # how many unique words to use (i.e num rows in embedding vector)

maxlen = 150 # max number of words in a question to use
my_list = [i for i in existing_df['text']]
my_words = " ".join(i for i in my_list)
words_length = len(set(my_words.split()))

words_length
max_features = words_length
## fill up the missing values

train_X = existing_df['text'].fillna("##").values



print("before tokenization")

print(train_X.shape)





## Tokenize the sentences

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(train_X))



train_X = tokenizer.texts_to_sequences(train_X)



print("after tokenization")

print(len(train_X))
train_X = pad_sequences(train_X, maxlen=maxlen)
gc.collect()
existing_df.head()
x_train = existing_df.drop(['sentiment', 'IDcustomer',  'ID_perfume', 'text'], axis=1)
x_train.shape
train_X.shape
import numpy as geek 

#geek.save('tokenisedfile', train_X) 
#np.savetxt('textembeddings.txt', train_X)
gc.collect()
x_train.shape, train_X.shape, existing_df['sentiment'].shape
np.random.seed(0)



indices = np.random.permutation(x_train.shape[0])

training_idx, test_idx = indices[:700000], indices[700000:]
x_train, x_test = x_train.iloc[training_idx,:], x_train.iloc[test_idx,:]



x_train_embed, x_test_embed = train_X[training_idx,:], train_X[test_idx,:]
x_train.shape, x_test.shape
target = existing_df['sentiment'].values
target
target_encoded = np.where(target>0.6, 1, 0)
target_encoded
y_train, y_test = target_encoded[training_idx], target_encoded[test_idx]
y_train.shape, y_test.shape
gc.collect()
x_train_embed.shape, x_train.shape
inp = Input(shape=(maxlen,))

x = Embedding(max_features, embed_size)(inp)

#lstm = Bidirectional(LSTM(200, dropout=0.2, recurrent_dropout=0.2))(x)

#x = Embedding(max_features, embed_size)(inp)



x = LSTM(256, return_sequences=True)(x)

#x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)

x = LSTM(64, return_sequences=True)(x)

x = Flatten()(x)



agei = Input(shape=(153,))

#agei = Dense(100)(agei)



conc = concatenate([x, agei])



drop = Dropout(0.2)(conc)

dens = Dense(100)(drop)

dens = Dense(1)(dens)

acti = Activation('sigmoid')(dens)



model = Model(inputs=[inp, agei], outputs=acti)

#optimizer = optimizers.Adam(lr=1e-4)

#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



model.compile(loss='binary_crossentropy', optimizer='sgd', metrics = ['acc'])
#SVG(model_to_dot(model).create(prog='dot', format='svg'))
model.fit([x_train_embed, x_train], y_train, validation_data=([x_test_embed, x_test], y_test),epochs=1,

          batch_size=1024, shuffle=True, verbose=1)
model.save('my_model.h5') 
model.save('my_model_new.hdf5') 
#predicted = model.predict([x_train_embed, x_train])
#predicted[:100]
#target[training_idx][:100]
#existing_df.head()
gc.collect()
#existing_df.to_csv("existing_df.csv", index = None)
gc.collect()
my_list = list(product(existing_df['IDcustomer'][:10], existing_df['ID_perfume'].unique()))

newdf = pd.DataFrame(data=my_list, columns=['IDcustomer','ID_perfume'])
newdf['IDcustomer'].nunique()
newdf['IDcustomer']
newdf['combo'] = newdf['IDcustomer'].apply(str) + " " + newdf['ID_perfume'].apply(str)

combination_existed['combo'] = combination_existed['IDcustomer'].apply(str) + " " + combination_existed['ID_perfume'].apply(str)
combination_existed['combo'].values
test_df = newdf.loc[~newdf['combo'].isin(combination_existed['combo'].values)]
del test_df['combo']
gc.collect()
del existing_df

del newdf

del combination_existed
gc.collect()
test_df = pd.merge(test_df, cust_df, on='IDcustomer', how='left')

test_df = pd.merge(test_df, perf_df, on='ID_perfume', how='left')
test_df.shape
test_X = test_df['text'].fillna("##").values



test_X = tokenizer.texts_to_sequences(test_X)



test_X = pad_sequences(test_X, maxlen=maxlen)
test_df.head()
x_test = test_df.drop(['IDcustomer',  'ID_perfume', 'text'], axis=1)
x_test.shape
test_X.shape
y_pred = model.predict([test_X, x_test])
y_pred[:5]
test_df['prediction'] = y_pred
test_df.head()
def get_recommendations(cust_id):

    results = test_df[test_df['IDcustomer']==cust_id][['ID_perfume', 'prediction']]

    return results.sort_values(by ='prediction' , ascending=False)[:10]

get_recommendations(1141379)
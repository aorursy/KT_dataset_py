from __future__ import division

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import math

import keras

import theano

from scipy import sparse

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from sklearn.naive_bayes import GaussianNB

from sklearn.feature_extraction.text import TfidfVectorizer

sns.set(color_codes=True)

df_wine = pd.read_csv('./../input/winemag-data-130k-v2.csv')

data = df_wine.drop_duplicates('description')

data = data[pd.notnull(data.price)]

df_wine=data[:10000]

df_wine.head()

from sklearn.feature_extraction.text import TfidfVectorizer

tokenize = lambda doc: doc.lower().split(" ")

sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)

tfidf_transformer = sklearn_tfidf.fit(df_wine['description'].values)

tfidf_out = tfidf_transformer.transform(df_wine['description'].values)

vocab = np.array(tfidf_transformer.get_feature_names())
sns.distplot(df_wine['points'].values)

plt.show()

df_wine['points'][df_wine['points'].values > 95].describe()
def largemean(data):

    calcout = []

    for x in range(500,data.shape[0],500):

        calcout.append(np.mean(data[x-500:x,:],0))

    out = np.mean(np.array(calcout),0)

    return out



tfidf_in = tfidf_out[np.argsort(df_wine['points'].values)]

points = np.sort(df_wine['points'].values)



data = tfidf_in[np.where(points > 95)].toarray()

data_op = tfidf_in[np.where(points <= 95)].toarray()

def tf_idf_relative(data,data_op):

    mean_up = largemean(data)

    mean_down = largemean(data_op)

    data_mean_relative = np.mean(data,0) - np.mean(data_op,0)

    data_argsort = np.argsort(data_mean_relative)[::-1]

    return data_argsort



data_argsort = tf_idf_relative(data,data_op)

top40 = vocab[data_argsort][:40]



print('top 40 words that describe a wine which will exceed 95 points')

for x in range(0,len(top40) , 4):

    print(top40[x],',', top40[x+1],',',top40[x+2],',',top40[x+3])
Y=[]

for pnt in points:

    if pnt <= 95:

        Y.append([0,1])

    else:

        Y.append([1,0])



Y_ = np.asarray(Y)

X_ = tfidf_in.toarray()

pct = int(0.85 * Y_.shape[0])

X_train = X_[:pct]

X_test = X_[pct:]

Y_train = Y_[:pct]

Y_test = Y_[pct:]


model = Sequential()

model.add(Dense(1000 , input_dim=X_train.shape[1], activation='softmax'))

model.add(Dropout(0.25))

model.add(Dense(1060, activation='softmax'))

model.add(Dropout(0.5))

model.add(Dense(Y_train.shape[1], activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam',  metrics=['accuracy'] )

#model.compile(loss='mean_squared_error', optimizer='adam',  metrics=['accuracy'] )

model.fit(X_train, Y_train, epochs=4, batch_size=100)



score = model.evaluate(X_test, Y_test, verbose=True)



trial_X = X_test[8].reshape(1,X_test[5].shape[0])

trial = model.predict(trial_X)



print( 'model accuracy:' , round(score[1],3) , '%')
data_cb = df_wine[df_wine['description'].str.contains('cheeseburger') == True]

data_cb_f = df_wine[df_wine['description'].str.contains('cheeseburger') == False][:5000]



tf_cb = tfidf_transformer.transform(data_cb['description'].values)

tf_cb_f = tfidf_transformer.transform(data_cb_f['description'].values)



data_mean_relative = np.mean(tf_cb,0) - np.mean(tf_cb_f ,0)

data_argsort = np.argsort(data_mean_relative)[0,::-1]



top = vocab[data_argsort][0][:60]

print

print('top 60 words that describe a wine that pairs well with a cheeseburger:')

for x in range(0,len(top) , 4):

    print(top[x],',', top[x+1],',',top[x+2],',',top[x+3])
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import os



from tqdm import tqdm

import math

from sklearn.model_selection import train_test_split

from sklearn import metrics



from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D

from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D, SpatialDropout1D

from tensorflow.keras.models import Model

from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers

from tensorflow.keras.optimizers import Adam, SGD, RMSprop

from tensorflow.keras.callbacks import EarlyStopping



import lightgbm as lgb

from sklearn.svm import SVC

from sklearn.model_selection import KFold
import nltk

from nltk.sentiment import vader

analyzer = vader.SentimentIntensityAnalyzer()



#!pip install vaderSentiment

#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
data = pd.read_csv("../input/tweets.csv")

data = data[["Sarcasm", "tweet"]]

#data.head(10)

print(data.tweet[0])
data['tweet'] = data['tweet'].str.replace('#sarcasm', '')

print(data.tweet[0])
#analyzer = SentimentIntensityAnalyzer()



final_list = []

for sent in data['tweet']:

    senti = analyzer.polarity_scores(sent)

    list_temp=[]

    for key, value in senti.items():

        temp = value

        list_temp.append(temp)

    final_list.append(list_temp)
temp_df = pd.DataFrame(final_list, columns=['compound','neg','neu','pos'], index=data.index)

data = pd.merge(data, temp_df, left_index=True,right_index=True)

data.head()
train_df, test_df = train_test_split(data, test_size=0.15, random_state=101)

train_df, val_df = train_test_split(train_df, test_size=0.10, random_state=101)

print("Train size:{}".format(train_df.shape))

print("Validation size:{}".format(val_df.shape))

print("Test size:{}".format(test_df.shape))
embed_size = 300 

max_features = 50000 

maxlen = 100 



## fill up the missing values

train_X = train_df["tweet"].fillna("_na_").values

val_X = val_df["tweet"].fillna("_na_").values

test_X = test_df["tweet"].fillna("_na_").values



## Tokenize the sentences

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(train_X))

train_X = tokenizer.texts_to_sequences(train_X)

val_X = tokenizer.texts_to_sequences(val_X)

test_X = tokenizer.texts_to_sequences(test_X)



## Pad the sentences 

train_X = pad_sequences(train_X, maxlen=maxlen)

val_X = pad_sequences(val_X, maxlen=maxlen)

test_X = pad_sequences(test_X, maxlen=maxlen)



## Get the target values

train_y = train_df['Sarcasm'].values

val_y = val_df['Sarcasm'].values
inp = Input(shape=(maxlen,))

x = Embedding(max_features, embed_size)(inp)

x = Conv1D(256, maxlen)(x)

x = GlobalMaxPool1D()(x)

x = Dense(64, activation="relu")(x)

x = Dropout(0.2)(x)

x = Dense(32, activation="relu")(x)

x = Dropout(0.2)(x)

x = Dense(16, activation="relu")(x)

x = Dropout(0.2)(x)

x = Dense(1, activation="sigmoid")(x)

model2 = Model(inputs=inp, outputs=x)

adam =  Adam(lr=0.0001,decay=0.00001)

model2.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])



print(model2.summary())
model2.fit(train_X, train_y, batch_size=512, epochs=10, validation_data=(val_X, val_y), 

           callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=15)])
y_pred2 = model2.predict([val_X], batch_size=512, verbose=1)

y_pred2 = y_pred2>0.4



from sklearn.metrics import accuracy_score, confusion_matrix

print("Accuracy Score: ", accuracy_score(val_y, y_pred2))

print("Confusion Matrix: \n", confusion_matrix(val_y, y_pred2))

print("F1 Score: ", metrics.f1_score(val_y, y_pred2))
data_X = data["tweet"].fillna("_na_").values

data_X = tokenizer.texts_to_sequences(data_X)

data_X = pad_sequences(data_X, maxlen=maxlen)



y_pred_data2 = model2.predict([data_X], batch_size=512, verbose=1)
d = pd.DataFrame(y_pred_data2,columns=['CNN'], index=data.index)



data = pd.merge(data, d, left_index=True,right_index=True)



data.head()
temp_X = data[['compound','neg','neu','pos','CNN']]

temp_y = data['Sarcasm']



X_train, X_test, y_train, y_test = train_test_split(temp_X,temp_y, test_size=0.33, random_state=101)
lgb_train = lgb.Dataset(X_train, y_train)

lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)



params = {'boosting_type': 'gbdt',

          'objective': 'binary',

          'metric': {'l2', 'l1'},

          'num_leaves': 100,

          'learning_rate': 0.1,

          'feature_fraction': 0.9,

          'bagging_fraction': 0.8,

          'bagging_freq': 5,

          'verbose': 1

         }



gbm = lgb.train(params,

                lgb_train,

                num_boost_round=500,

                valid_sets=lgb_eval,

                early_stopping_rounds=15)
y_pred_lgb = gbm.predict(X_test, num_iteration=gbm.best_iteration)

y_pred_lgb = y_pred_lgb>0.4



s=metrics.f1_score(y_test, y_pred_lgb)

ss=accuracy_score(y_test, y_pred_lgb)

print("Accuracy Score: ", accuracy_score(y_test, y_pred_lgb))

print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred_lgb))

print("F1 Score: ", metrics.f1_score(y_test, y_pred_lgb))
svc = SVC(gamma='auto')



cv = KFold(n_splits=10, random_state=42, shuffle=True)

scores = []

i = 1

for train_index, test_index in cv.split(temp_X):

    

    X_train, X_test =  temp_X.values[train_index], temp_X.values[test_index]

    y_train, y_test = temp_y[train_index], temp_y[test_index]

    

    svc.fit(X_train, y_train)

    print("Iteration: ",i," - Score = ",svc.score(X_test, y_test).round(3))

    scores.append(svc.score(X_test, y_test))

    i+=1
y_pred_svc = svc.predict(temp_X)



sss=metrics.f1_score(temp_y, y_pred_svc)

ssss=accuracy_score(temp_y, y_pred_svc)

print("Accuracy Score: ", accuracy_score(temp_y, y_pred_svc))

print("Confusion Matrix: \n", confusion_matrix(temp_y, y_pred_svc))

print("F1 Score: ", metrics.f1_score(temp_y, y_pred_svc))
score=[s,sss]



label=['lightGBM','SVM']

index = np.arange(len(label))

plt.bar(index, score)

plt.xlabel('Algorithms', fontsize=12)

plt.ylabel('score', fontsize=12)

plt.xticks(index, label, fontsize=10, rotation=0)

plt.title('F1-Score plot')

plt.show()
score=[ss,ssss]



label=['lightGBM','SVM']

index = np.arange(len(label))

plt.bar(index, score)

plt.xlabel('Algorithms', fontsize=12)

plt.ylabel('accuracy', fontsize=12)

plt.xticks(index, label, fontsize=10, rotation=0)

plt.title('Accuracy Plot')

plt.show()
       
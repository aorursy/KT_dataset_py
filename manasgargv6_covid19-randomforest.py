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
from datetime import datetime



import pandas as pd

import numpy as np

from nltk.stem import WordNetLemmatizer

from nltk import tokenize

import re

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.naive_bayes import BernoulliNB, MultinomialNB

from sklearn import metrics

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB



import os

import sys

import numpy as np

import keras

from keras.preprocessing.text import Tokenizer

from keras.utils import to_categorical

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Activation, Conv2D, Input, Embedding, Reshape, MaxPool2D, Concatenate, Flatten, Dropout, Dense, Conv1D

from keras.layers import MaxPool1D

from keras.models import Model

from keras.callbacks import ModelCheckpoint

from keras.optimizers import Adam



train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv")

test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/test.csv")







print(train.columns)

print(test.columns)



print(len(train['Date'].unique()))

print(len(test['Date'].unique()))



print(train.info())

#print(test.summary)



total = train.isnull().sum()

print(total)



train['Long'].describe()



print(train['Lat'].isnull().sum())



train['Lat']=train['Lat'].ffill(axis=0)



train['Long']= train['Long'].ffill(axis=0)

test['Lat']=test['Lat'].ffill(axis=0)

test['Long']=test['Long'].ffill(axis=0)



train['Province/State'] = train['Province/State'].fillna(train['Country/Region'])

test['Province/State'] = test['Province/State'].fillna(test['Country/Region'])



total = train.isnull().sum()

print(total)

total = test.isnull().sum()

print(total)



data=[train,test]

ll=[]

y1 = train['Date'].unique()

y2 = test['Date'].unique()



print(len(y1))



in_first = set(y1)

in_second = set(y2)



in_second_but_not_in_first = in_second - in_first



result = y1

print(result)



l1=[]

for i in y1:

  l1.append(i)



for i in y2:

  l1.append(i)

  

print(len(l1))



tt = set(l1)

print(len(tt))

pp=[]

for i in tt:

  pp.append(i)

pp.sort(key=lambda date: datetime.strptime(date, "%Y-%m-%d"))

print(pp)

#print(tt)

dates={}

k=0

for i in pp:

  dates[i]=k

  k=k+1



print(dates)



data = [train,test]

for dataset in data:

    dataset['Date1'] = dataset['Date'].map(dates)



print(train['Date1'])



import nltk

nltk.download('wordnet')



l1=[]

for i in train['Country/Region']:

  l1.append(i)



for i in train['Province/State']:

  l1.append(i)



tokenizer  = Tokenizer()

tokenizer.fit_on_texts(l1)

sequences =  tokenizer.texts_to_sequences(l1)







train['CP']=""

train['S']=""

for i in range(0,len(train)):

  train['CP'][i] = sequences[i]



print(i)

for j in range(0,len(train)):

  i = i+1

  train['S'][j] = sequences[i]



l2=[]

for i in test['Country/Region']:

  l2.append(i)



for i in test['Province/State']:

  l2.append(i)





tokenizer.fit_on_texts(l2)

sequences1=  tokenizer.texts_to_sequences(l2)



test['CP']=""

test['S']=""

for i in range(0,len(test)):

  test['CP'][i] = sequences1[i]



print(i)

for j in range(0,len(test)):

  i = i+1

  test['S'][j] = sequences1[i]



t1 = set(l1)

countries={}

k=0

for i in t1:

  countries[i]=k

  k=k+1



data = [train,test]

for dataset in data:

    dataset['CP2'] = dataset['Country/Region'].map(countries)

    dataset['S2'] = dataset['Province/State'].map(countries)





train_df = train

test_df = test



print(train_df.columns)



X_train=train_df.drop("Id",axis=1).copy()

X_train=X_train.drop("Province/State",axis=1)

X_train=X_train.drop("Country/Region",axis=1)

X_train=X_train.drop("Date",axis=1)

X_train=X_train.drop("CP",axis=1)

X_train=X_train.drop("S",axis=1)



X_test=test_df.drop("ForecastId",axis=1).copy()

X_test=X_test.drop("Province/State",axis=1)

X_test=X_test.drop("Country/Region",axis=1)

X_test=X_test.drop("Date",axis=1)

X_test=X_test.drop("CP",axis=1)

X_test=X_test.drop("S",axis=1)



print(X_train.columns)

print(X_test.columns)







Y_train = X_train['Fatalities']

X_train = X_train.drop("Fatalities",axis=1)

Y_train1 = X_train['ConfirmedCases']

X_train = X_train.drop("ConfirmedCases",axis=1)



print(X_train.columns)







random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train1)



Y_prediction = random_forest.predict(X_test)



random_forest.score(X_train, Y_train1)

acc_random_forest = round(random_forest.score(X_train, Y_train1) * 100, 2)

print(X_train.columns)

print(X_test.columns)
X_test['CC']=""

for i in range(0,len(Y_prediction)):

  X_test['CC'][i] = Y_prediction[i]



X_train['CC']=""

for i in range(0,len(Y_train1)):

  X_train['CC'][i] = Y_train1[i]



print(X_train.columns)

print(X_test.columns)
XX = X_train[['Lat','Long','Date1','CC','CP2','S2']]

print(XX)
XX1 = X_test[['Lat','Long','Date1','CC','CP2','S2']]
from nltk.stem import WordNetLemmatizer

from nltk import tokenize

import re

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.naive_bayes import BernoulliNB, MultinomialNB

from sklearn import metrics

from sklearn.decomposition import TruncatedSVD

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import Normalizer

from sklearn.naive_bayes import GaussianNB

from sklearn.decomposition import PCA

'''

pca = PCA(n_components=2)

X_train_count =pca.fit_transform(X_train) 

X_test_counts = pca.transform(X_test)

'''



svd = TruncatedSVD(n_components=2)

normalizer = Normalizer(copy=False)

lsa = make_pipeline(svd, normalizer)



X_train_count = lsa.fit_transform(XX)

X_test_counts = lsa.transform(XX1)
t = set(train['Fatalities'])
print(len(t))
random_forest = RandomForestClassifier(n_estimators=190)

random_forest.fit(X_train_count, Y_train)



Y_prediction = random_forest.predict(X_test_counts)



random_forest.score(X_train_count, Y_train)

acc_random_forest = round(random_forest.score(X_train_count, Y_train) * 100, 2)



print(acc_random_forest)



import csv

fields = ['ForecastId','ConfirmedCases','Fatalities'] 

with open("submission.csv", 'w') as csvfile: 

    # creating a csv writer object 

    csvwriter = csv.writer(csvfile) 

      

    # writing the fields 

    csvwriter.writerow(fields) 

      

    # writing the data rows 

    for i in range(0,len(Y_prediction)):

      csvwriter.writerow([test['ForecastId'][i],X_test['CC'][i],Y_prediction[i]])

      
!pip install anvil-uplink
import numpy as np

import pandas as pd

import csv

from collections import defaultdict

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import nltk

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix,classification_report



#for pipeline

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline



import anvil.server



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_excel("../input/raw-data/raw_data.xlsx")
data
data = data.fillna(method='ffill')
data['Disease'] = data['Disease'].str.split("^",expand = True)[0]

data['Symptom'] = data['Symptom'].str.split("^",expand = True)[0]
data['Disease'] = data['Disease'].str.split("_",expand = True)[1]

data['Symptom'] = data['Symptom'].str.split("_",expand = True)[1]
data_new = data.copy()
def renew_data(data, col1, col2):

    i = 1

    while i<data.shape[0]:

        if data[col1].iloc[i]==data[col1].iloc[i-1]:

            data[col2].iloc[i] = data[col2].iloc[i-1]+" "+data[col2].iloc[i]

        i = i +1

    return data
renew_data(data_new,'Disease','Symptom')
data_new.groupby('Disease').describe()
cv = CountVectorizer()
X = data_new['Symptom']

Y = data_new['Disease']
X = cv.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2,random_state=101)
#nb = MultinomialNB()
#nb.fit(X_train,y_train)
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
predictions = rfc.predict(X_train)

print(confusion_matrix(y_train,predictions))

print('\n')

print(classification_report(y_train,predictions))
test_predictions = rfc.predict(X_test)

print(confusion_matrix(y_test,test_predictions))

print('\n')

print(classification_report(y_test,test_predictions))
'''z = []

t = input().lower()

z.append(t)

z = pd.DataFrame(z, columns=['ABC'])

z = z['ABC']

z = cv.transform(z)

z_pred = rfc.predict(z)

z_pred[0]'''
anvil.server.connect('DZU26OT5HWL75CS5ACYEJPMK-7IV5XSTFJEP5BWVW')

@anvil.server.callable

def classify_disease(sym):

    z = []

    z.append(sym)

    z = pd.DataFrame(z, columns=['ABC'])

    z = z['ABC']

    z = cv.transform(z)

    z_pred = rfc.predict(z)

    return z_pred[0]
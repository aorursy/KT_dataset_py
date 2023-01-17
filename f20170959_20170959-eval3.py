import numpy as np

import pandas as pd

import sklearn

import matplotlib.pyplot as plt



%matplotlib inline
from sklearn.cluster import KMeans, AgglomerativeClustering

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
data = pd.read_csv("train.csv")

data.drop('custId', axis=1, inplace=True)

categorical_features = ['Internet','gender', 'SeniorCitizen', 'Married', 'Children', 'TVConnection', 'Channel1', 'Channel2', 'Channel3', 'Channel4','Channel5', 'Channel6','HighSpeed', 'AddedServices', 'Subscription', 'PaymentMethod']



y = data['Satisfied']

data.drop(['Satisfied'], axis=1, inplace=True)
def preprocess(data):

    data.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    data['TotalCharges'] = data['TotalCharges'].astype(float)

    data = data.apply(LabelEncoder().fit_transform)

    data.fillna(value = data.mean(), inplace=True)

    data = sklearn.preprocessing.normalize(data)

    data = np.array(data)

    data = data.reshape((data.shape[0],data.shape[1]))

    return data
data = preprocess(data)
data
x_train, x_test, y_train, y_test = train_test_split(data, y,test_size=0.10,random_state=42)
aggl = AgglomerativeClustering()
aggl.fit(x_train)

pred = aggl.fit_predict(data)
len(pred)
acc = sum((1-pred) == y)

print(acc/len(y))
df_test = pd.read_csv('test.csv')

df_test.head()
test_cust_id = df_test['custId']

df_test.drop('custId', axis=1, inplace=True)

categorical_features = ['Internet','gender', 'SeniorCitizen', 'Married', 'Children', 'TVConnection', 'Channel1', 'Channel2', 'Channel3', 'Channel4','Channel5', 'Channel6','HighSpeed', 'AddedServices', 'Subscription', 'PaymentMethod']
df_test = preprocess(df_test)
df_test

len(df_test)
pred_test = aggl.fit_predict(df_test)
print(len(pred_test))
submission = pd.concat([test_cust_id, pd.Series(1-pred_test)], axis=1)

submission.columns = ['custId', 'Satisfied']

submission.to_csv('submission.csv', index=False)
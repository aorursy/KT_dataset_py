#Dipayan Deb 2019H1030015G

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE 

%matplotlib inline
df = pd.read_csv("/kaggle/input/eval-lab-3-f464/train.csv")
df.head()
df.isnull().sum()
df.info()
df.corr()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
categorical_feature_mask = df.dtypes==object

categorical_cols = df.columns[categorical_feature_mask].tolist()

df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))
df.head()
co = [

 'SeniorCitizen',

    'Married',

 'Children',

 'TVConnection',

'Channel1',

 'Channel2',

 'Channel3',

 'Channel4',

 'Channel5',

 'Channel6',

    'Internet',

'HighSpeed',

 'AddedServices',

 'Subscription',

 'tenure',

'PaymentMethod',

 'MonthlyCharges',

]

df.columns.tolist()
X = df[co]

y = df["Satisfied"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)



sm = SMOTE(random_state = 2) 

X_train, y_train = sm.fit_sample(X_train, y_train) 
from sklearn.cluster import KMeans

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

#scaler = MinMaxScaler()

#scaler.fit(X_train)



scaler = StandardScaler()

scaler.fit(X_train)





X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)



kmeans = KMeans(n_clusters=2)

kmeans.fit(X_train)

y_pred = kmeans.predict(X_test)
A1 = metrics.accuracy_score(y_test,y_pred)

print("Accuracy for KMeans")

print("Accuracy_1: ",A1)

y_pred = pd.DataFrame(y_pred)

y_pred.replace(0,2,inplace=True)

y_pred.replace(1,0,inplace=True)

y_pred.replace(2,1,inplace=True)

A2 = metrics.accuracy_score(y_test,y_pred)

print("Accuracy_2: ",A2)
kf = pd.read_csv("/kaggle/input/eval-lab-3-f464/test.csv")

categorical_feature_mask = kf.dtypes==object

categorical_cols = kf.columns[categorical_feature_mask].tolist()

kf[categorical_cols] = kf[categorical_cols].apply(lambda col: le.fit_transform(col))

myid1 = kf["custId"].copy()

X1 = kf[co]

X1 = scaler.transform(X1)







y_tt = kmeans.predict(X1)

y_tt = pd.DataFrame(y_tt)

if A2>A1:

    y_tt.replace(0,2,inplace=True)

    y_tt.replace(1,0,inplace=True)

    y_tt.replace(2,1,inplace=True)



    

ans = pd.concat([myid1,y_tt],axis=1)

ans.columns=["custId","Satisfied"]

ans = ans.set_index("custId")

ans.to_csv("boo.csv")
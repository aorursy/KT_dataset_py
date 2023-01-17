# common python imports

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import pickle as pkl

from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score,precision_score,recall_score,confusion_matrix
# reading dataset

df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")

df.head()
df.shape
# Checking how balanced the dataset

print("Non-Fraud Cases Count : ",df[df['Class']==0].count()[0])

print("Fraud Cases Count : ",df[df['Class']==1].count()[0])
# MinMaxScaler function

def scaling(df,scaler=None):

    if scaler==None:

        scaler = MinMaxScaler()

        scaler.fit(df)

        df = scaler.transform(df)

        pkl.dump(scaler,open("fraud_scaler.pkl",'wb'))

    else:

        df = scaler.transform(df)

    return df
# splitting the dependent & independent variables

y = df.Class

X = df.drop(columns=['Class'],axis=1)
# handling unbalanced dataset problems

from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)

X_resampled, y_resampled = ros.fit_resample(X, y)
X_resampled = pd.DataFrame(X_resampled,columns=X.columns)

y_resampled = pd.DataFrame(y_resampled,columns=['Class'])
X_resampled.shape
X_train,X_test,y_train,y_test = train_test_split(X_resampled,y_resampled,test_size=0.33,random_state=42)
X_train = scaling(X_train)

X_train
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train,y_train)
X_test = scaling(X_test,pkl.load(open("fraud_scaler.pkl",'rb')))
y_pred = logreg.predict(X_test)
# the problem with unbalanced dataset - fraud is not very frequent (less patterns to detect fraud) so we have done over sampling

confusion_matrix(y_test,y_pred)
print("F1 Score : ",f1_score(y_test,y_pred))
import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df = pd.read_csv('../input/detecting-parkinson/parkinsons.data')
df.head()
features=df.loc[:,df.columns!='status'].values[:,1:]
labels=df.loc[:,'status'].values
print(labels[labels==1].shape[0], labels[labels==0].shape[0])
scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(features)
y=labels 
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)
model=XGBClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(accuracy_score(y_test, y_pred)*100)
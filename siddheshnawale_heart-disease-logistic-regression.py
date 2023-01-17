import numpy as np

import pandas as pd

import seaborn as sb

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LogisticRegression

import warnings

warnings.filterwarnings('ignore')

from sklearn import metrics

import missingno as mo
df_final= pd.read_csv('../input/MLChallenge-2/final.csv')

df_test= pd.read_csv('../input/MLChallenge-2/Test.csv')
df_final.head()
df_test
df_final.shape
df_final.columns
df_final.skew()
df_final.kurtosis()


#scaling the data 

#Scaling

from sklearn.preprocessing import MinMaxScaler

#data=features_df_new.drop(labels='diagnosis',axis=1)

#target=features_df_new['diagnosis']

scaler = MinMaxScaler()

data = pd.DataFrame(scaler.fit_transform(df_final), columns=df_final.columns)

#dataaa = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
data_test=pd.DataFrame(scaler.fit_transform(df_test), columns=df_test.columns)
data
data_test
from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LogisticRegression 
data_test.drop(labels='ID',axis=1,inplace=True)
data.drop(labels='ID',axis=1,inplace=True)
X=data.drop(labels='target',axis=1)

y=data[['target']]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
loreg=LogisticRegression()
loreg.fit(X_train,y_train)
prediction=loreg.predict(data_test)
prediction=pd.DataFrame(prediction)
prediction.shape
loreg.score(X_train,y_train)
y_pred_train=loreg.predict(X_train)
from sklearn.metrics import accuracy_score

accuracy_score(y_train,y_pred_train)
loreg.fit(X_test,y_test)
y_pred_test=loreg.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_score(y_pred_test, y_test)
prediction.to_csv('submission.csv')
prediction
data_test
submission=df_test.join(prediction)
submission.rename(columns={0:"target"})
submission.to_csv('submission.csv')
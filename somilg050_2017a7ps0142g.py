import warnings

warnings.filterwarnings("ignore")

import sklearn

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score, ShuffleSplit

from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler
df = pd.read_csv('../input/eval-lab-1-f464-v2/train.csv')

df.info()
df.drop('id', axis=1, inplace=True)

df.head()
df.fillna(value=df.mean(),inplace=True)
X=pd.get_dummies(data=df,columns=['type'])

y=df['rating']

X.head()

X.drop(columns=['rating'],inplace=True)

X=(X-X.mean())/X.std()
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_squared_error

# MAXV, MAXI, MAXJ = 100,0,0

# for i in range(20,33):

#     for j in range(30,50):

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=40)

gbrt=RandomForestClassifier(n_estimators=116, n_jobs=-1, random_state=29)

gbrt.fit(X_train, y_train) 

y_pred=gbrt.predict(X_test)

score = mean_squared_error(y_test,y_pred)

print(score)

#         if score < MAXV:

#             MAXV = score

#             MAXI = i

#             MAXJ = j

# print(MAXV, MAXI, MAXJ)
df2 = pd.DataFrame(data=y_pred,columns=["something"])

df2.something.value_counts()
df1 = pd.read_csv('../input/eval-lab-1-f464-v2/test.csv')

df1.fillna(value=df1.mean(),inplace=True)

df1=pd.get_dummies(data=df1,columns=['type'])

df1.drop(columns=['id'],axis=1, inplace=True)

df1 = (df1 - df1.mean())/df1.std()

df1.head()

var = gbrt.predict(df1)

df2 = pd.read_csv('../input/eval-lab-1-f464-v2/test.csv')

df4=pd.DataFrame(index=df2['id'])

df4['rating']=var

df4.to_csv('../s1.csv')

df2 = pd.DataFrame(data=var,columns=["something"])

df2.something.value_counts()
pd.DataFrame(data=gbrt.feature_importances_).plot()
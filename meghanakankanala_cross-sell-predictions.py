# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
test=pd.read_csv("/kaggle/input/janatahack-crosssell-prediction/test.csv")
train=pd.read_csv("/kaggle/input/janatahack-crosssell-prediction/train.csv")
print(train.shape)
print(test.shape)
data=pd.concat([train,test],axis=0)
data.drop("Response",axis=1,inplace=True)
train.head()
print(data.shape)
data.isnull().sum()
data.head()
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
data.Gender=encoder.fit_transform(data.Gender)
data.Vehicle_Damage=encoder.fit_transform(data.Vehicle_Damage)
data.head()
data["Vehicle_Age"].unique()
vehicle_age={"< 1 Year": 0,'1-2 Year':1,'> 2 Years':2}
data["Vehicle_Age"]=data["Vehicle_Age"].replace(vehicle_age)
data["Vehicle_Age"].unique()
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
cor=data.corr()
plt.figure(figsize=(10,10))
sns.heatmap(cor,annot=True)

column=data.columns
from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler()
sdata=scale.fit_transform(data)
Data=pd.DataFrame(sdata,columns=column)
Data.head()
Train=Data.iloc[:381109,:]
Test=Data.iloc[381109: ,:]
Train["Response"]=train["Response"]
train["Response"].value_counts()
cor_re=Train.corr()
plt.figure(figsize=(10,10))
sns.heatmap(cor_re,annot=True)
y=Train.Response
# mod_=data.drop(columns=["Response"],axis=1,inplace=False)
# total_data=pd.concat([mod_train ,Test])
data=data.drop(columns=["Age","Vehicle_Damage","id"],axis=1,inplace=False)

Mod_train=data.iloc[:381109,:]
Mod_test=data.iloc[381109:,:]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(Mod_train,y,test_size=0.2,random_state=1)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import roc_auc_score,classification_report
print(roc_auc_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
Mod_train["Response"]=train["Response"]
from sklearn.utils import resample
min_data=Mod_train[Mod_train["Response"]==1]
maj_data=Mod_train[Mod_train["Response"]==0]
mod_min_data=resample(min_data,n_samples=334399,replace=True)
mod_data=pd.concat([maj_data,mod_min_data])
mod_data.shape
Y=mod_data.Response
X=mod_data.drop("Response",axis=1)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)
print(roc_auc_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))
from sklearn.model_selection import cross_val_score
score=cross_val_score(model,X,Y,cv=10,scoring="roc_auc")
print(score.mean())


predictions=model.predict(Mod_test)
result=pd.DataFrame(test["id"],columns=["id","Response"])
result["Response"]=predictions
result.to_csv("sub.csv",index=0)

from sklearn.metrics import SCORERS
print(SCORERS.keys())

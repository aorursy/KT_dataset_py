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
data=pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv')

datas=pd.read_csv('../input/pubg-finish-placement-prediction/test_V2.csv')

dat=datas.Id

data.head()

dat
data.isnull().sum()
datas.isnull().sum()


data.fillna(method='ffill',axis=0,inplace=True)
data.isnull().sum().sum()
data.drop(['Id','matchId','groupId'],axis=1,inplace=True)

datas.drop(['Id','matchId','groupId'],axis=1,inplace=True)

data= pd.get_dummies(data, columns=["matchType"])

datas= pd.get_dummies(datas,columns=['matchType'])

import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(40,40))

cor = data.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
cor_target = abs(cor["winPlacePerc"])

relevant_features = cor_target[cor_target>0.5]

relevant_features
print(data[["boosts","walkDistance"]].corr())

print(datas[['killPlace','weaponsAcquired']].corr())

print(data[["boosts","weaponsAcquired"]].corr())
X_train=data.loc[:,['killPlace']]

Y_train=data.iloc[:,-1]

X_test=datas.loc[:,['killPlace']]
from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(X_train,Y_train)

regressor.predict(X_test)

import xgboost as xgb

model = xgb.XGBRegressor()

model.fit(X_train,Y_train)

model.predict(X_test)
X_train=data.loc[:,['killPlace','weaponsAcquired']]

Y_train=data.iloc[:,-1]

X_test=datas.loc[:,['killPlace','weaponsAcquired']]

import xgboost as xgb

model = xgb.XGBRegressor()

model.fit(X_train,Y_train)

result=model.predict(X_test)
final = pd.DataFrame({'Id': dat,

                       'winPlacePerc':result })

final.to_csv('submission.csv', index=False)
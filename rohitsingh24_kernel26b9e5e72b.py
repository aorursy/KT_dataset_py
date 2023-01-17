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
import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd

housetest = pd.read_csv("../input/housetest.csv")

housetrain = pd.read_csv("../input/housetrain.csv")

housetrain
housetrain.info()
housetrain.hist(bins=80,figsize=(20,20))

plt.show
housetrain["LotArea"].plot(kind = 'hist',bins = 250,figsize = (6,6))

plt.title("LotArea")

plt.xlabel("LotArea")

plt.ylabel("Frequency")

plt.show()
housetrain['LotFrontage'].value_counts()
housetrain['LotFrontage'].fillna(housetrain['LotFrontage'].mode()[0],inplace=True)
housetrain['MasVnrType'].value_counts()
housetrain['MasVnrType'].fillna(housetrain['MasVnrType'].mode()[0],inplace=True)
housetrain['MasVnrArea'].value_counts()
housetrain['MasVnrArea'].fillna(housetrain['MasVnrArea'].mode()[0],inplace=True)
housetrain['BsmtQual'].value_counts()
housetrain['BsmtQual'].fillna(housetrain['BsmtQual'].mode()[0],inplace=True)
housetrain['BsmtCond'].value_counts()
housetrain['BsmtCond'].fillna(housetrain['BsmtCond'].mode()[0],inplace=True)
housetrain['BsmtExposure'].value_counts()
housetrain['BsmtExposure'].fillna(housetrain['BsmtExposure'].mode()[0],inplace=True)
housetrain['BsmtFinType1'].value_counts()
housetrain['BsmtFinType1'].fillna(housetrain['BsmtFinType1'].mode()[0],inplace=True)
housetrain['BsmtFinType2'].value_counts()
housetrain['BsmtFinType2'].fillna(housetrain['BsmtFinType2'].mode()[0],inplace=True)
housetrain['Electrical'].value_counts()
housetrain['Electrical'].fillna(housetrain['Electrical'].mode()[0],inplace=True)
housetrain['GarageType'].value_counts()
housetrain['GarageType'].fillna(housetrain['GarageType'].mode()[0],inplace=True)
housetrain['GarageYrBlt'].value_counts()
housetrain['GarageYrBlt'].fillna(housetrain['GarageYrBlt'].mode()[0],inplace=True)
housetrain['GarageFinish'].value_counts()
housetrain['GarageFinish'].fillna(housetrain['GarageFinish'].mode()[0],inplace=True)
housetrain['GarageQual'].value_counts()
housetrain['GarageQual'].fillna(housetrain['GarageQual'].mode()[0],inplace=True)
housetrain['GarageCond'].value_counts()
housetrain['GarageCond'].fillna(housetrain['GarageCond'].mode()[0],inplace=True)
housetrain.info()
X=housetrain.drop(['Id','Alley','PoolQC','Fence','MiscFeature','FireplaceQu',"SalePrice"],axis=1)
X.shape
Y=housetrain['SalePrice']
X.info()
plt.figure(figsize=(15,15))

sns.heatmap(X.corr(),annot=True,linewidths=0.05,fmt='.2f',cmap='magma')

X=pd.get_dummies(X)

X.head()
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25)
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])

X_test[X_test.columns] = scaler.transform(X_test[X_test.columns])

X_train.head()



X_train.shape
from sklearn.ensemble import RandomForestRegressor

rgr=RandomForestRegressor()

rgr.fit(X_train,Y_train)
rgr.score(X_test,Y_test)
import xgboost as xgb

xg = xgb.XGBRegressor()

xg.fit(X_train,Y_train)
xg.score(X_test,Y_test)
#from sklearn.matrics import mean_squared_error,r2_score,mean_absolute_error



#print('Mean Absolute Error:',means_absolute_error(Y_test,))
housetest.head()
housetest.columns
housetest.drop(['Id','Alley','PoolQC','Fence','MiscFeature','FireplaceQu'],axis=1,inplace=True)
housetest.info()
def count(val):

    r=housetest[val].value_counts()

    print(r)

def fill(value):

    housetest[value].fillna(housetest[value].mode()[0],inplace=True)
col=housetest.columns.tolist()
for i in col:

    count(i)

    fill(i)
housetest.info()
x=pd.get_dummies(housetest)
from sklearn.preprocessing import MinMaxScaler

#x[X_t.columns] = scaler.transform(X_test[X_test.columns])

x[x.columns] = scaler.fit_transform(x[x.columns])

x.head()

missing_cols = set( X_train.columns ) - set( x.columns )

missing_cols
for c in missing_cols:

    x[c] = 0
x.shape
#x[x.columns] = scaler.transform(x[x.columns])
y_predict=rgr.predict(X_test)

y_predict.shape

#Y_test.shape
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error

import numpy as np



print('Mean Absolute Error:', mean_absolute_error(Y_test, y_predict))  

print('Mean Squared Error:', mean_squared_error(Y_test, y_predict))  

print('Root Mean Squared Error:', np.sqrt(mean_squared_error(Y_test, y_predict)))
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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
train=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
print(train.select_dtypes(include=object).columns)
train.describe()
from sklearn.preprocessing import LabelEncoder



for i in range(train.shape[1]):

    if train.iloc[:,i].dtypes == object:

        lbl = LabelEncoder()

        lbl.fit(list(train.iloc[:,i].values) + list(test.iloc[:,i].values))

        train.iloc[:,i] = lbl.transform(list(train.iloc[:,i].values))

        test.iloc[:,i] = lbl.transform(list(test.iloc[:,i].values))
train.info()
train.head()
# keep ID for submission

train_ID = train['Id']

test_ID = test['Id']



# split data for training

y_train = train['SalePrice']

X_train = train.drop(['Id','SalePrice'], axis=1)

X_test = test.drop('Id', axis=1)



# dealing with missing data

Xmat = pd.concat([X_train, X_test])

Xmat = Xmat.drop(['LotFrontage','MasVnrArea','GarageYrBlt'], axis=1)

Xmat = Xmat.fillna(Xmat.median())
Xmat['TotalSF']=Xmat['TotalBsmtSF']+Xmat['1stFlrSF']+Xmat['2ndFlrSF']
y_train=np.log(y_train)

ax=sns.distplot(y_train)

plt.show()
X_train.shape
X_train.info()
X_train['LotFrontage']=X_train['LotFrontage'].apply('int64')

X_train['MasVnrArea']=X_train['MasVnrArea'].apply('int64')

X_train['GarageYrBlt']=X_train['GarageYrBlt'].apply('int64')
X_train.drop(X_train.columns[np.isnan(X_train).any()], axis=1)
X_train.shape
import missingno as msno

msno.matrix(df=train, figsize=(20,14), color=(0.5,0,0))
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=79, max_features='auto')

rf.fit(X_train, y_train)

print('Training done using Random Forest')



ranking = np.argsort(-rf.feature_importances_)

f, ax = plt.subplots(figsize=(11, 9))

sns.barplot(x=rf.feature_importances_[ranking], y=X_train.columns.values[ranking], orient='h')

ax.set_xlabel("feature importance")

plt.tight_layout()

plt.show()
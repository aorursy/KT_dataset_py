import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder  ###for encode a categorical values

from sklearn.model_selection import train_test_split  ## for spliting the data

from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMRegressor  
train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

train.head()
train.isnull().any()
miss_col=[col for col in train.columns if train[col].isnull().any()]

print(miss_col)
for col in miss_col:

    train[col]=train[col].fillna(train[col].mode()[0])
train.isnull().sum()
train.info()
LE=LabelEncoder()

for col in train.select_dtypes(include=['object']):

    train[col]=LE.fit_transform(train[col])
train.head()
#standardizing data

saleprice_scaled = StandardScaler().fit_transform(train['SalePrice'][:,np.newaxis])

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
# Adding total sqfootage feature 

train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']

print(train)

train.drop(['TotalBsmtSF','1stFlrSF','2ndFlrSF'],axis=1)
test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

test.isnull().sum()
miss_test=[col for col in test.columns if test[col].isnull().any()]

print(miss_test)
for col in miss_test:

    test[col]=test[col].fillna(test[col].mode()[0])
test.head()
for col in test.select_dtypes(include=['object']):

    test[col]=LE.fit_transform(test[col])



test.head()
# Adding total sqfootage feature 

test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']

print(test)

test.drop(['TotalBsmtSF','1stFlrSF','2ndFlrSF'],axis=1)
X_train=train.drop(["SalePrice"],axis=1)

Y_train=train["SalePrice"]

print(X_train)
from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(X_train , Y_train ,test_size = 0.1,random_state = 0)
lightgbm = LGBMRegressor(objective='regression', 

                                       num_leaves=8,

                                       learning_rate=0.0385, 

                                       n_estimators=3500,

                                       max_bin=200, 

                                       bagging_fraction=0.75,

                                       bagging_freq=5, 

                                       bagging_seed=7,

                                       feature_fraction=0.2,

                                       feature_fraction_seed=7,

                                       verbose= 0,

                                       )
lightgbm.fit(X_train,Y_train)

lightgbm.score(x_test,y_test)
 

prediction = lightgbm.predict(test)
sub=pd.DataFrame()

sub['Id']=test['Id']

sub['SalePrice']=prediction

sub.to_csv('submission.csv',index=False)
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
import pandas as pd

import numpy as np
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df_train.describe()
df_train = df_train.set_index('Id',drop=True)
y_train = df_train.pop('SalePrice')

X_train = df_train
(X_train.isna().sum(0) / X_train.shape[0]).sort_values(ascending=False)[:20]
int2str = ['MSSubClass', 'OverallQual', 'OverallCond']

nan2cat = ['Alley', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']

def preprocess(X):

    X[int2str] = X[int2str].astype(str)

    X[nan2cat] = X[nan2cat].fillna('NA')

    return X

X_train[int2str] = X_train[int2str].astype(str)

X_train[nan2cat] = X_train[nan2cat].fillna('NA')
from sklearn.impute import SimpleImputer



cat_imputer = SimpleImputer(strategy = 'most_frequent')

num_imputer = SimpleImputer(strategy = 'mean')



num_features = [c for c in X_train.columns if X_train[c].dtype in (int,float)]

cat_features = [c for c in X_train.columns if not c in num_features]



num_imputer.fit(X_train[num_features])

cat_imputer.fit(X_train[cat_features])



X_train_num = num_imputer.transform(X_train[num_features])

X_train_cat = cat_imputer.transform(X_train[cat_features])
from sklearn.preprocessing import OneHotEncoder



ohe = OneHotEncoder(handle_unknown='ignore')

ohe.fit(X_train_cat)

X_train_cat_01 = ohe.transform(X_train_cat)
from scipy import sparse as sp



X_concated = sp.hstack([

    X_train_num,

    X_train_cat_01

]).toarray()
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

scaler.fit(X_concated)

X_scaled = scaler.transform(X_concated)
from sklearn.ensemble import RandomForestRegressor



rf = RandomForestRegressor(n_estimators=100)

rf.fit(X_scaled, y_train)

p_train = rf.predict(X_scaled)
from sklearn.metrics import mean_squared_error



mean_squared_error(np.log(y_train),np.log(p_train))
from sklearn.model_selection import cross_val_score



def scoring(model, X,y):

    p = model.predict(X)

    return mean_squared_error(np.log(y), np.log(p))



cross_val_score(rf, X_scaled, y_train, scoring=scoring, cv=5)
X_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
X_test.describe()
X_test = X_test.set_index('Id',drop=True)

X_test[int2str] = X_test[int2str].astype(str)

X_test[nan2cat] = X_test[nan2cat].fillna('NA')



X_test_num = num_imputer.transform(X_test[num_features])

X_test_cat = cat_imputer.transform(X_test[cat_features])



X_test_cat_01 = ohe.transform(X_test_cat)



X_test_concated = sp.hstack([

    X_test_num,

    X_test_cat_01

]).toarray()



X_test_scaled = scaler.transform(X_test_concated)



p_test = rf.predict(X_test_scaled)
output = pd.DataFrame({'Id':X_test.index, 'SalePrice':p_test})

output.to_csv('/kaggle/working/my_submission.csv', index=False)
print(open('/kaggle/working/my_submission.csv').read())
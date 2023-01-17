import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn

from sklearn import ensemble

from copy import copy

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

from scipy.stats import skew  # for some statistics

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

from sklearn.model_selection import KFold, cross_val_score

import warnings

# Ignore useless warnings

warnings.filterwarnings("ignore")



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.index=train.Id

train = train.drop(columns=['Id'])

test.index=test.Id

test = test.drop(columns=['Id'])



train = train[train.GrLivArea < 4500]

Y = train['SalePrice']

all_features = list(train.columns)

all_features.remove('SalePrice')

X = train[all_features]
X2 = test.copy()

data = pd.concat([X,X2])
## 填充缺失值

missing_data = data[data.columns[data.isnull().sum()>0]]

missing_obj = missing_data.select_dtypes(include=np.object).columns

for item in list(missing_data.select_dtypes(include=np.object).columns):

    data[item] = data[item].fillna('without')

data['LotFrontage'] =data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))    

data = data.fillna(0) 

skew_features = data[list(missing_data.select_dtypes(exclude=np.object).columns)].apply(

    lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]

skew_index = high_skew.index



for i in skew_index:

    data[i] = boxcox1p(data[i], boxcox_normmax(data[i] + 1))

data = data.drop(columns = ['Utilities', 'Street', 'PoolQC'])



data['YrBltAndRemod']=data['YearBuilt']+data['YearRemodAdd']

data['TotalSF']=data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']



data['Total_sqr_footage'] = (data['BsmtFinSF1'] + data['BsmtFinSF2'] +

                                 data['1stFlrSF'] + data['2ndFlrSF'])



data['Total_Bathrooms'] = (data['FullBath'] + (0.5 * data['HalfBath']) +

                               data['BsmtFullBath'] + (0.5 * data['BsmtHalfBath']))



data['Total_porch_sf'] = (data['OpenPorchSF'] + data['3SsnPorch'] +

                              data['EnclosedPorch'] + data['ScreenPorch'] +

                              data['WoodDeckSF'])

data['haspool'] = data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

data['has2ndfloor'] = data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

data['hasgarage'] = data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

data['hasbsmt'] = data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

data['hasfireplace'] = data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

data = pd.get_dummies(data)

X_data = data.iloc[:len(train)]

X_pred = data.iloc[len(train):]

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_data,Y,test_size=0.33)
#确定最终模型

final_params = {'learning_rate': 0.01, 'n_estimators': 3000, 'gamma': 0, 'min_child_weight':0,

          'colsample_bytree': 0.7, 'objective': 'reg:linear','nthread': -1, 'reg_lambda': 10,

          'scale_pos_weight': 1, 'seed': 27,'reg_alpha': 6e-05, 'random_state': 42}

model = XGBRegressor(**final_params)

model.fit(X_data,Y)
result = pd.DataFrame(model.predict(X_pred),index =X_pred.index,columns=['SalePrice'])

result.to_csv('my_submition.csv')
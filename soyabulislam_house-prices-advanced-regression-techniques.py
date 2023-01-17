import os

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

import seaborn as sns

import matplotlib.pyplot as plt
train_data= pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train_data.head()

train_data.describe()



train_data.isnull().sum()
train_data.info()
train_data=train_data.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu'], axis=1)

train_data.info()
import matplotlib.pyplot as plt

train_data.hist(bins=50, figsize=(20,15))

plt.show()
test.head()
test.info()
test=test.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu'], axis=1)

test
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)

val = train_data["SalePrice"]

train_data.drop(['SalePrice'], axis=1, inplace=True)
#cols_with_missing = [col for col in train_data.columns if train_data[col].isnull().any()] 

#train_data.drop(cols_with_missing, axis=1, inplace=True)

#test.drop(cols_with_missing, axis=1, inplace=True)
test.info()
features=['MSSubClass','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','1stFlrSF',

 '2ndFlrSF','LowQualFinSF','GrLivArea','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd',

 'Fireplaces','WoodDeckSF','EnclosedPorch','OpenPorchSF','3SsnPorch','ScreenPorch','PoolArea',

 'MiscVal', 'MoSold','YrSold']



train_data= train_data[features].copy()

test_data = test[features].copy()
X_train, X_val, y_train, y_val= train_test_split(train_data, val, random_state=40, train_size=0.8)
train_data['LotArea'].value_counts().plot(kind='bar')

plt.title('Plot area')

plt.xlabel('LotArea')

plt.ylabel('Count')

sns.despine



X_train.head()

X_train.info()
from sklearn.preprocessing import OneHotEncoder

object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]

high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))





OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols ]))

OH_cols_val = pd.DataFrame(OH_encoder.transform(X_val[low_cardinality_cols ]))





OH_cols_train.index = X_train.index

OH_cols_val.index = X_val.index



num_X_train = X_train.drop(object_cols, axis=1)

num_X_val = X_val.drop(object_cols, axis=1)





X_train = pd.concat([num_X_train, OH_cols_train], axis=1)

X_val = pd.concat([num_X_val, OH_cols_val], axis=1)
from sklearn import ensemble

elr= ensemble.GradientBoostingRegressor()

model=elr.fit(X_train, y_train)



model.score(X_val, y_val)

from sklearn.linear_model import LinearRegression

lr= LinearRegression()

model1=lr.fit(X_train, y_train)

model1.score(X_val, y_val)

from sklearn.ensemble import RandomForestRegressor

rfr= RandomForestRegressor()

model2= rfr.fit(X_train, y_train)

model2.score(X_val, y_val)


from sklearn.linear_model import  Ridge, Lasso

ridge = Ridge(alpha = 1)  # sets alpha to a default value as baseline  

model3=ridge.fit(X_train, y_train)

model3.score(X_val, y_val)
lasso= Lasso(alpha = .001)  # sets alpha to a default value as baseline  

model4= lasso.fit(X_train, y_train)

model4.score(X_val, y_val)
from sklearn.model_selection import RandomizedSearchCV,KFold

from xgboost import XGBRegressor

params = {

        'min_child_weight': [1, 5, 10],

        'colsample_bytree': np.arange(0.3,0.5,0.1),

        'colsample_bylevel': np.arange(0.3,0.5,0.1),

        'max_depth': range(6,10),

        'n_estimators': range(100,1000)

        }

xgb = XGBRegressor(learning_rate=0.04,gamma = 0,subsample = 1,silent=True, nthread=-1,n_jobs=-1)

rand = RandomizedSearchCV(xgb, n_iter=20,refit ='MSLE', param_distributions=params, n_jobs=-1,verbose=2, random_state=50,return_train_score = True)

model5= rand.fit(X_train, y_train)

model5.score(X_val, y_val)
object_col = [col for col in test_data.columns if test_data[col].dtype == "object"]

test_d=test_data.drop(object_col, axis=1)

test_d.head()
cols_with_missing = [col for col in test_d.columns

                     if test_d[col].isnull().any()]





test_x= test_d.drop(cols_with_missing, axis=1)

test_x.head()
test_x.info()

prediction=model5.predict(X_val)

prediction
from sklearn.metrics import mean_squared_error

print ('MSE is: \n', mean_squared_error(y_val, prediction))

final_prediction= model5.predict(test_x)

final_prediction



output = pd.DataFrame({'Id': test["Id"],

                       'SalePrice': final_prediction})

output.to_csv('submission.csv', index=False)
output

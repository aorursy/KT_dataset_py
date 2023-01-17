import pandas as pd

import numpy as np

import seaborn as sns
train = pd.read_csv('/kaggle/input/predict-the-housing-price/train.csv')

test= pd.read_csv('/kaggle/input/predict-the-housing-price/Test.csv')
train.head(3)
test.head(3)
train.isnull().sum()
test.isnull().sum()
import matplotlib.pyplot as plt
plt.subplots(figsize=(20,30))

sns.heatmap(train.corr())
# droping columns which are having more number of null values

drop_columns = ['FireplaceQu','PoolQC','Fence','MiscFeature','BsmtUnfSF']

train.drop(drop_columns, axis = 1, inplace = True)

test.drop(drop_columns, axis = 1, inplace = True)
fill_col = ['Alley','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',

            'GarageType','GarageFinish','GarageCond']

for col in train[fill_col]:

    train[col] = train[col].fillna('None')

for col in test[fill_col]:

    test[col] = test[col].fillna('None')
#filling null values by mean 

train['LotFrontage'].fillna(value=train['LotFrontage'].mean(),inplace=True)

test['LotFrontage'].fillna(value=test['LotFrontage'].mean(),inplace=True)
train['GarageYrBlt'].fillna(value=train['GarageYrBlt'].mean(),inplace=True)

test['GarageYrBlt'].fillna(value=test['GarageYrBlt'].mean(),inplace=True)
train['MasVnrArea'].fillna(value=train['MasVnrArea'].mean(),inplace=True)

test['MasVnrArea'].fillna(value=test['MasVnrArea'].mean(),inplace=True)
# filling nullvalues by using mode

train['GarageQual'].fillna(value=(train['GarageQual'].mode()[0]),inplace=True)

test['GarageQual'].fillna(value=(test['GarageQual'].mode()[0]),inplace=True)
test['Electrical'].fillna(value=(test['Electrical'].mode()[0]),inplace=True)
train
test
# assigning categorical variables.

category = []

for i in train.columns:

    if train[i].dtype == "O":

        category.append(i)
category
train[category]
test[category]
for i in train.isnull().sum():

    if i > 0:

        print(i)
for i in test.isnull().sum():

    if i > 0:

        print(i)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
temp1 = train[category].apply(LabelEncoder().fit_transform)
temp2 = test[category].apply(LabelEncoder().fit_transform)
train.drop(category,axis = 1,inplace =True)

test.drop(category,axis = 1,inplace =True)
trainfinal = train.join(temp1)

testfinal = test.join(temp2)
trainfinal
testfinal
trainfinal.drop('Id',axis=1,inplace=True)

testfinal.drop('Id',axis=1,inplace=True)
from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import RFE

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

import statsmodels.api as sm
X = trainfinal.drop('SalePrice',axis = 1)

y = trainfinal[['SalePrice']]
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=.7,random_state=42)

lr = LinearRegression()
lr = LinearRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

print("RMSE ->",np.sqrt(mean_squared_error(y_test, y_pred)))
### Using Rfe

scaler = MinMaxScaler()

X_train1 = scaler.fit_transform(X_train)

y_train1 = scaler.fit_transform(y_train)

rfe = RFE(lr, 1)

rfe.fit(X_train1,y_train1)
rfe.support_
X_train.columns[rfe.support_]
X = trainfinal['OverallQual'].values.reshape(-1,1)

y = trainfinal[['SalePrice']].values.reshape(-1,1)
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=.7,random_state=42)
lr = LinearRegression()
lr.fit(X_train,y_train)

lr.predict(X_test)

print("RMSE ->",np.sqrt(mean_squared_error(y_test, y_pred)))
X = trainfinal.drop('SalePrice',axis = 1)

y = trainfinal[['SalePrice']]

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=.7,random_state=42)
import xgboost as xgb
xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =42, nthread = -1)
xgb.fit(X_train,y_train)
y_pred = xgb.predict(X_test)
print("RMSE =>",np.sqrt(mean_squared_error(y_test, y_pred)))
## Testing on Test Dataset

y_predtest = xgb.predict(testfinal)
temp = pd.DataFrame()

temp['Id'] = test['Id']

temp['SalePrice'] = y_predtest
temp
temp.to_csv('solution1', index=False)
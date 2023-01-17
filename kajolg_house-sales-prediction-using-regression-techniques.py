# Important Libraries

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

%matplotlib inline



from sklearn.metrics import mean_squared_log_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

pd.set_option('max_columns', 200)
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.shape
test.shape
train.head()
test.head()
train.columns
# to know the missing values in columns

print(train.info())

print('**'* 50)

print(test.info())
# Now removing the columns with large number of missing values from training set

drop_columns = ['FireplaceQu','PoolQC','Fence','MiscFeature','BsmtUnfSF']

train.drop(drop_columns, axis = 1, inplace = True)

test.drop(drop_columns, axis = 1, inplace = True)
train['LotFrontage'].fillna(train['LotFrontage'].median(), inplace = True)

train['MasVnrArea'].fillna(train['MasVnrArea'].median(), inplace = True)
fill_col = ['Alley','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',

            'GarageType','GarageFinish','GarageCond']

both_col = [train,test]

for col in both_col:

    col[fill_col] = col[fill_col].fillna('None')
colfil = ['LotFrontage','MasVnrArea','BsmtFinSF1','BsmtFinSF2','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageCars', 

            'GarageArea']

for coll in colfil:

    test[coll].fillna(test[coll].median(), inplace = True)
both_col = [train, test]

for col in both_col:

    col['YrBltAndRemod'] = col['YearBuilt'] + col['YearRemodAdd']

    col['TotalSF'] = col['TotalBsmtSF'] + col['1stFlrSF'] + col['2ndFlrSF']

    col['Total_sqr_footage'] = (col['BsmtFinSF1'] + col['BsmtFinSF2'] +

                                 col['1stFlrSF'] + col['2ndFlrSF'])



    col['Total_Bathrooms'] = (col['FullBath'] + (0.5 * col['HalfBath']) +

                               col['BsmtFullBath'] + (0.5 *col['BsmtHalfBath']))



    col['Total_porch_sf'] = (col['OpenPorchSF'] + col['3SsnPorch'] +

                              col['EnclosedPorch'] + col['ScreenPorch'] +

                              col['WoodDeckSF'])
both_col = [train,test]

for col in both_col:

    col['haspool'] = col['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

    col['has2ndfloor'] = col['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

    col['hasgarage'] = col['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

    col['hasbsmt'] = col['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

    col['hasfireplace'] = col['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
train.shape
test.shape
drop_col = ['Exterior2nd','GarageYrBlt','Condition2','RoofMatl','Electrical','HouseStyle','Exterior1st',

            'Heating','GarageQual','Utilities','MSZoning','Functional','KitchenQual']

train.drop(drop_col, axis = 1,inplace = True)

test.drop(drop_col, axis = 1,inplace = True)
train.shape
test.shape
train.info()
test.info()
# to know the correlation between the each columns in dataset

plt.figure(figsize=(30,10))

sns.heatmap(test.corr(),cmap='coolwarm',annot = True)

plt.show()
x_t = train.drop(['SalePrice'], axis = 1)

y_t = train['SalePrice']
# to convert a categorical into one hot encoding

hot_one= pd.get_dummies(x_t)
hot_one.shape
x_train ,x_test ,y_train ,y_test = train_test_split(hot_one, y_t, test_size = 0.3, random_state = 0)
model = RandomForestRegressor(n_estimators = 300, random_state = 0)

model.fit(x_train,y_train)
dtr_pred = model.predict(x_test)
from IPython.display import Image

Image("../input/kaggle/Screenshot-2019-05-16-at-6.43.11-PM-850x262.png")


print('RMSE:', np.sqrt(mean_squared_log_error(y_test, dtr_pred)))
# how the model predicts against the actual known value

plt.figure(figsize=(15,8))

plt.scatter(y_test,dtr_pred, c='green')

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.show()
hot_two = pd.get_dummies(test)
hot_two.shape
hot_two.head()
test_prediction = model.predict(hot_two)
test_pred = pd.DataFrame(test_prediction, columns=['SalePrice'])
test_pred.head()
from sklearn import ensemble

modell = ensemble.GradientBoostingRegressor(n_estimators = 3000, max_depth = 5,max_features='sqrt', min_samples_split = 10,

          learning_rate = 0.005,loss = 'huber',min_samples_leaf=15,random_state =42)

modell.fit(x_train, y_train)
pred = modell.predict(x_test)
print('RMSE:', np.sqrt(mean_squared_log_error(y_test, pred)))
# how the model predicts against the actual known value

plt.figure(figsize=(15,8))

plt.scatter(y_test,pred, c='red')

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.show()
test_pred_2 = modell.predict(hot_two)
test_pred = pd.DataFrame(test_pred_2, columns=['SalePrice'])
test_pred.head()
# Gradient Boosting has the low error as compare with Random Forest Regression

out = pd.DataFrame({'Id': hot_two['Id'], 'SalePrice': test_pred_2})

out.to_csv('submission.csv', index=False,header=True)
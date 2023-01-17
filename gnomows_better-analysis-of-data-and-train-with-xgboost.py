# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
train.head()
train_label = train['SalePrice']
numeric_columns = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 
                  'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',  
                  '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'FullBath',
                  'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',
                   'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',  '3SsnPorch', 'ScreenPorch', 
                   'PoolArea', 'MiscVal', 'MoSold', 'YrSold']
index = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 
        'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
        'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond',
        'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'Heating', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType',
        'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive','PoolQC', 'Fence', 'MiscFeature',
        'SaleType', 'SaleCondition']
i = 0
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
df = train[[index[i]]]
df["B"] = df[index[i]].astype('category')
i=i+1
df['B']
## 6 types for SalesCondition: [Abnorml, AdjLand, Alloca, Family, Normal, Partial]
num = train[numeric_columns]
index = train[index]
from sklearn.model_selection import train_test_split
# Train_test split with 25% test size
train_data, test_data, train_labels, test_labels = train_test_split(num, 
                                                                    train_label, 
                                                                    test_size=0.20)
import xgboost as xgb
import numpy as np
# Flatten columns
train_labels = np.ravel(train_labels)
test_labels = np.ravel(test_labels)

# Create DMatrix for xgboost
D_train = xgb.DMatrix(data=train_data, silent=1, nthread=-1, label =train_labels)
D_test  = xgb.DMatrix(data=test_data,  silent=1, nthread=-1, label =test_labels)
param = {'silent' : 1,
         'learning_rate' : 0.03,
         'max_depth': 10,
         'tree_method': 'exact',
         'objective': 'reg:linear'
         }

n_rounds = 300

watch_list = [(D_train, 'train'), (D_test, 'eval')]
bst = xgb.train(param, D_train, n_rounds, watch_list, early_stopping_rounds = 15)
pred = bst.predict( D_test )
predictions = [np.around(value) for value in pred]
from sklearn.metrics import accuracy_score, precision_score
accuracy = accuracy_score(test_labels, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
df2 = test_data
df2.loc[:,('pred')] = list(predictions)
df2.loc[:,('real')] = list(test_labels)
df2.groupby("pred").agg("count")
df2.groupby("real").agg("count")
xgb.plot_importance(bst)
# Using model to test data
test_df = pd.read_csv('../input/test.csv')
test_df
num = test_df[numeric_columns]
D_test  = xgb.DMatrix(data=num,  silent=1, nthread=-1)
pred = bst.predict( D_test )
predictions = [np.around(value) for value in pred]
test_df['SalePrice'] = predictions
test_df
submission = test_df[['Id', 'SalePrice']]
submission.to_csv('submission.csv', index = False)
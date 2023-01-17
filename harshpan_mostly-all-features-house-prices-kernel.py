# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
df = pd.read_csv('../input/train.csv')

# Any results you write to the current directory are saved as output.
def convert_to_numerical(dframe, list_of_cat_columns):
    for col in list_of_cat_columns:
        dframe[col] = dframe[col].astype('category')
        dframe[col] = dframe[col].cat.codes
        
    return dframe
    
def set_type_to_cat_for_test(train_dframe, test_dframe, list_of_cat_columns):
    for col in list_of_cat_columns:
        test_dframe[col] = test_dframe[col].astype('category')
        test_dframe[col].cat.set_categories(train_dframe[col].unique().tolist(), ordered=True, inplace=True)
        test_dframe[col] = test_dframe[col].cat.codes
        
    return test_dframe

def fill_na(dframe):
    dframe.LotFrontage = dframe.groupby('Neighborhood').LotFrontage.apply(lambda x: x.fillna(x.median()))
    dframe.Alley = dframe.Alley.fillna('NA')
    dframe.MasVnrType = dframe.MasVnrType.fillna('None')
    dframe.MasVnrArea = dframe.MasVnrArea.fillna(0)
    dframe.BsmtQual = dframe.BsmtQual.fillna('NA')
    dframe.BsmtCond = dframe.BsmtCond.fillna('NA')
    dframe.BsmtExposure = dframe.BsmtExposure.fillna('NA')
    dframe.BsmtFinType1 = dframe.BsmtFinType1.fillna('NA')
    dframe.BsmtFinType2 = dframe.BsmtFinType2.fillna('NA')

    dframe.TotalBsmtSF = dframe.TotalBsmtSF.fillna(0)
    dframe.BsmtFullBath = dframe.BsmtFullBath.fillna(0)
    dframe.BsmtHalfBath = dframe.BsmtHalfBath.fillna(0)
    dframe.GarageCars = dframe.GarageCars.fillna(0)
    dframe.GarageArea = dframe.GarageArea.fillna(0)

    dframe.Electrical = dframe.Electrical.fillna('SBrkr')
    dframe.FireplaceQu = dframe.FireplaceQu.fillna('NA')
    dframe.GarageType = dframe.GarageType.fillna('NA')
    dframe.GarageYrBlt = dframe.GarageYrBlt.fillna(-1)
    dframe.GarageFinish = dframe.GarageFinish.fillna('NA')
    dframe.GarageCond = dframe.GarageCond.fillna('NA')
    dframe.GarageQual = dframe.GarageQual.fillna('NA')
    dframe.PoolQC = dframe.PoolQC.fillna('NA')
    dframe.Fence = dframe.Fence.fillna('NA')
    dframe = dframe.drop(['MiscFeature', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF'], axis=1)
    return dframe

def get_categorical_data(dframe):
    list_of_cat_columns = list(dframe.select_dtypes(include=['object']).columns)
    return convert_to_numerical(dframe, list_of_cat_columns)

def get_categorical_data_for_test(train_dframe, test_dframe):
    list_of_cat_columns = list(train_dframe.select_dtypes(include=['object']).columns)        
    return set_type_to_cat_for_test(train_dframe, test_dframe, list_of_cat_columns)
df.head()
df = fill_na(df)
df.head()
df_orig = df.copy()
df = get_categorical_data(df)
df
columns_not_included = ['SalePrice', '1stFlrSF', 'Id']
feature_columns = df.columns.difference(columns_not_included)

df_train_X = df[feature_columns]
df_train_y = np.log(df.SalePrice)
model = Ridge()
parameters = {
    'alpha': [2, 1, 0.1, 0.01],
    'random_state': [42],
    'max_iter': [300, 500, 800]
}

clf = GridSearchCV(model, parameters, cv=5, scoring='neg_mean_squared_error', return_train_score=True, verbose=1)
clf.fit(X=df_train_X, y=df_train_y)
clf.best_estimator_
math.sqrt(-clf.best_score_)
# ordered_df = pd.DataFrame(list(zip(model.coef_, feature_columns.tolist()))).sort_values(0, ascending=False)
# ordered_df

# ordered_df.plot.barh()
# fig = plt.gcf()
# fig.set_size_inches(18.5, 10.5)
test_df = pd.read_csv('../input/test.csv')

test_df = fill_na(test_df)

test_df = get_categorical_data_for_test(df_orig, test_df)
test_df_orig = test_df.copy()
test_df = test_df[feature_columns]

test_predicted = clf.best_estimator_.predict(test_df)

pd.DataFrame({'Id': test_df_orig['Id'], 'SalePrice': np.exp(test_predicted)}).to_csv('submission.csv', index=False)
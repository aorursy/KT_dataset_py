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
# Other useful packages to load

import matplotlib.pyplot as plt
# save filepaths to variables for easier access

train_file_path = '../input/train.csv'

test_file_path = '../input/test.csv'

sample_submission_filepath = '../input/sample_submission.csv'
# read the data and store in DataFrames

train_data = pd.read_csv(train_file_path)

test_data = pd.read_csv(test_file_path)

sample_submission_data = pd.read_csv(sample_submission_filepath)
# print a summary of the training data

train_description = train_data.describe()

train_description
# Note to self: DataFrames are accessed first by column, then by row.

train_description['Id']['count']
# DataFrame columns can be accessed with dot notation

train_description.Id['count']
# Get the column names

train_data.columns
train_data.info()
test_data.info()
train_y = train_data.SalePrice

train_data.drop('SalePrice', axis=1, inplace=True)
train_y.head()
train_data.head()
train_ids = train_data.Id

train_data.drop('Id', axis=1, inplace=True)

test_ids = test_data.Id

test_data.drop('Id', axis=1, inplace=True)
# This may become a preprocessing step for test data, but for now I'm just exploring

train_data_numeric = train_data.select_dtypes(include=[np.number])

train_data_categorical = train_data.drop(train_data_numeric.columns, axis=1)
train_data_numeric.head()
plt.scatter(train_data_numeric.MSSubClass, train_y)

plt.xlabel('MSSubClass')

plt.ylabel('SalePrice')
train_data_numeric = train_data.select_dtypes(include=[np.number]).drop('MSSubClass', axis=1)

train_data_categorical = train_data.drop(train_data_numeric.columns, axis=1)
train_data_categorical.head()
category_counts = []

for feature in train_data_categorical.columns:

    number_of_categories = len(train_data_categorical[feature].value_counts())

    category_counts.append((feature, number_of_categories))

category_counts.sort(key = lambda x : x[1], reverse=True)

print(category_counts)
print('Alley:')

print(train_data_categorical.Alley.value_counts())
print('MasVnrType:')

print(train_data_categorical.MasVnrType.value_counts())
train_data[train_data['MasVnrArea'].isna()]
print('BsmtQual:')

print(train_data_categorical.BsmtQual.value_counts())
train_data[train_data['BsmtQual'].isna()].TotalBsmtSF.value_counts()
print('BsmtCond:')

print(train_data_categorical.BsmtCond.value_counts())
train_data[train_data['BsmtCond'].isna()].TotalBsmtSF.value_counts()
print('BsmtExposure:')

print(train_data_categorical.BsmtExposure.value_counts())
train_data[train_data['BsmtExposure']=='No'].TotalBsmtSF.hist(bins=20)

plt.show()
train_data[train_data['BsmtExposure'].isna()].TotalBsmtSF.value_counts()
print('BsmtFinType2:')

print(train_data_categorical.BsmtFinType2.value_counts())
print(train_data[train_data['BsmtFinSF2']==0].BsmtFinType2.value_counts())
print('Electrical:')

print(train_data_categorical.Electrical.value_counts())
print('FireplaceQu:')

print(train_data_categorical.FireplaceQu.value_counts())
train_data[train_data['FireplaceQu'].isna()].Fireplaces.value_counts()
print('GarageType:')

print(train_data_categorical.GarageType.value_counts())
train_data[train_data['GarageType'].isna()].GarageArea.value_counts()
print('GarageFinish:')

print(train_data_categorical.GarageFinish.value_counts())
train_data[train_data['GarageFinish'].isna()].GarageArea.value_counts()
train_data[train_data['GarageFinish']=='Unf'].GarageArea.hist(bins=20)

plt.show()
print('GarageQual:')

print(train_data_categorical.GarageQual.value_counts())
print('GarageCond:')

print(train_data_categorical.GarageCond.value_counts())
print('PoolQC:')

print(train_data_categorical.PoolQC.value_counts())
print('Fence:')

print(train_data_categorical.Fence.value_counts())
print('MiscFeature:')

print(train_data_categorical.MiscFeature.value_counts())
train_data_numeric.head()
train_data_numeric.hist(bins=20, figsize=(20,15))

plt.show()
train_data_numeric_withSalePrice = pd.read_csv(train_file_path).drop(train_data_categorical.columns, axis=1).drop('Id', axis=1)

train_data_numeric_withSalePrice['LogSalePrice'] = np.log(train_data_numeric_withSalePrice['SalePrice'])

corr_matrix = train_data_numeric_withSalePrice.corr()
corr_matrix['LogSalePrice'].abs().sort_values(ascending=False)
for colname in train_data_numeric_withSalePrice.columns:

    train_data_numeric_withSalePrice.plot(kind='scatter', x=colname, y='LogSalePrice')
from pandas.plotting import scatter_matrix



attributes = ["GrLivArea", "GarageCars", "GarageArea", "TotalBsmtSF", "1stFlrSF"]

scatter_matrix(train_data_numeric_withSalePrice[attributes], figsize=(12, 8))
for df in (train_data, test_data):

    df['Alley'].fillna('None', inplace=True)

    df['Fence'].fillna('None', inplace=True)

    df['MiscFeature'].fillna('None', inplace=True)

    

    df['BsmtFullBath'].fillna(0, inplace=True)

    df['BsmtHalfBath'].fillna(0, inplace=True)

    

    df['KitchenQual'].fillna(train_data['KitchenQual'].mode()[0], inplace=True)

    df['Functional'].fillna(train_data['Functional'].mode()[0], inplace=True)

    df['SaleType'].fillna(train_data['SaleType'].mode()[0], inplace=True)

    df['Utilities'].fillna(train_data['Utilities'].mode()[0], inplace=True)

    df['Exterior1st'].fillna(train_data['Exterior1st'].mode()[0], inplace=True)

    df['Exterior2nd'].fillna(train_data['Exterior2nd'].mode()[0], inplace=True)

    df['Electrical'].fillna(train_data['Electrical'].mode()[0], inplace=True)

    df['Utilities'].fillna(train_data['Utilities'].mode()[0], inplace=True)

    df['Exterior1st'].fillna(train_data['Exterior1st'].mode()[0], inplace=True)

    df['Exterior2nd'].fillna(train_data['Exterior2nd'].mode()[0], inplace=True)

    

    df['TotalBsmtSF'].fillna(df['BsmtFinSF1']+df['BsmtFinSF2']+df['BsmtUnfSF'], inplace=True)

    df['TotalBsmtSF'].fillna(df['BsmtFinSF1']+df['BsmtFinSF2'], inplace=True)

    df['TotalBsmtSF'].fillna(df['BsmtFinSF1']+df['BsmtUnfSF'], inplace=True)

    df['TotalBsmtSF'].fillna(df['BsmtFinSF2']+df['BsmtUnfSF'], inplace=True)

    df['TotalBsmtSF'].fillna(df['BsmtFinSF1'], inplace=True)

    df['TotalBsmtSF'].fillna(df['BsmtFinSF2'], inplace=True)

    df['TotalBsmtSF'].fillna(df['BsmtUnfSF'], inplace=True)

    df['TotalBsmtSF'].fillna(0, inplace=True)

    df[df['TotalBsmtSF']!=0]['BsmtQual'].fillna(train_data[train_data['TotalBsmtSF']!=0]['BsmtQual'].mode()[0], inplace=True)

    df['BsmtQual'].fillna('None', inplace=True)

    df[df['TotalBsmtSF']!=0]['BsmtCond'].fillna(train_data[train_data['TotalBsmtSF']!=0]['BsmtCond'].mode()[0], inplace=True)

    df['BsmtCond'].fillna('None', inplace=True)

    df[df['TotalBsmtSF']!=0]['BsmtExposure'].fillna(train_data[train_data['TotalBsmtSF']!=0]['BsmtExposure'].mode()[0], inplace=True)

    df['BsmtExposure'].fillna('None', inplace=True)

    

    df['BsmtFinSF1'].fillna(0, inplace=True)

    df[df['BsmtFinSF1']!=0]['BsmtFinType1'].fillna(train_data[train_data['BsmtFinSF1']!=0]['BsmtFinType1'].mode()[0], inplace=True)

    df['BsmtFinType1'].fillna('Unf', inplace=True)

    

    df['BsmtFinSF2'].fillna(0, inplace=True)

    df[df['BsmtFinSF2']!=0]['BsmtFinType2'].fillna(train_data[train_data['BsmtFinSF2']!=0]['BsmtFinType2'].mode()[0], inplace=True)

    df['BsmtFinType2'].fillna('Unf', inplace=True)

    

    df['BsmtUnfSF'].fillna(df['TotalBsmtSF']-df['BsmtFinSF1']-df['BsmtFinSF2'], inplace=True)

    df[df['BsmtUnfSF']<0]['BsmtUnfSF'] = 0

    

    df['MasVnrArea'].fillna(0, inplace=True)

    df[df['MasVnrArea']!=0]['MasVnrType'].fillna(train_data[train_data['MasVnrArea']!=0]['MasVnrType'].mode()[0], inplace=True)

    df['MasVnrType'].fillna('None', inplace=True)

    

    df['GarageArea'].fillna(0, inplace=True)

    df[df['GarageArea']!=0]['GarageType'].fillna(train_data[train_data['GarageArea']!=0]['GarageType'].mode()[0], inplace=True)

    df['GarageType'].fillna('None', inplace=True)

    df[df['GarageArea']!=0]['GarageFinish'].fillna(train_data[train_data['GarageArea']!=0]['GarageFinish'].mode()[0], inplace=True)

    df['GarageFinish'].fillna('None', inplace=True)

    df[df['GarageArea']!=0]['GarageQual'].fillna(train_data[train_data['GarageArea']!=0]['GarageQual'].mode()[0], inplace=True)

    df['GarageQual'].fillna('None', inplace=True)

    df[df['GarageArea']!=0]['GarageCond'].fillna(train_data[train_data['GarageArea']!=0]['GarageCond'].mode()[0], inplace=True)

    df['GarageCond'].fillna('None', inplace=True)

    

    df['GarageYrBlt'].fillna(df['YearBuilt'], inplace=True)

    

    df['Fireplaces'].fillna(0, inplace=True)

    df[df['Fireplaces']!=0]['FireplaceQu'].fillna(train_data[train_data['Fireplaces']!=0]['FireplaceQu'].mode()[0], inplace=True)

    df['FireplaceQu'].fillna('None', inplace=True)

    

    df['PoolArea'].fillna(0, inplace=True)

    df[df['PoolArea']!=0]['PoolQC'].fillna(train_data[train_data['PoolArea']!=0]['PoolQC'].mode()[0], inplace=True)

    df['PoolQC'].fillna('None', inplace=True)
train_data.info()
from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import RobustScaler, OneHotEncoder

from sklearn.impute import SimpleImputer



num_cols = train_data_numeric.columns

cat_cols = train_data_categorical.columns



num_pipeline = Pipeline([

    ('num_imputer', SimpleImputer(strategy='median')),

    ('num_scaler', RobustScaler())

])



cat_pipeline = Pipeline([

    ('cat_nan_filler', SimpleImputer(strategy='constant', fill_value='None')),

    ('cat_onehot', OneHotEncoder(handle_unknown='ignore'))

])



preprocessor_pipeline = ColumnTransformer([

    ('num_pipeline', num_pipeline, num_cols),

    ('cat_pipeline', cat_pipeline, cat_cols)

])
preprocessed_train_data = preprocessor_pipeline.fit_transform(train_data)

preprocessed_test_data = preprocessor_pipeline.transform(test_data)
preprocessed_train_data.shape, preprocessed_test_data.shape
# Separate a validation set:

# (Hands-On Machine Learning with Scikit-Learn & TensorFlow, pg. 50 of my copy)

# Compute a hash of each instance's identifier,

# keep only the last byte of the hash,

# and put the instance in the val set if the value of that byte is < val_ratio*256.



val_ratio = 0.2

import hashlib

val_set_mask = train_ids.apply(lambda id : hashlib.md5(np.int64(id)).digest()[-1] < val_ratio * 256)
val_data = train_data.loc[val_set_mask]

train_data_noval = train_data.loc[~val_set_mask]
(len(val_data), len(train_data_noval))
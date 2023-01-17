# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
house_filepath='../input/house-prices-advanced-regression-techniques/train.csv'
df = pd.read_csv(house_filepath, index_col='Id')
df_test_filepath='../input/house-prices-advanced-regression-techniques/test.csv'
X_test=pd.read_csv(df_test_filepath, index_col='Id')
print(df.shape)
print(X_test.shape)
df
cat_cols = [col for col in df.columns if df[col].dtype == 'object']
len(cat_cols)
bad_cols = [ col for col in cat_cols if set(df[col].unique()) != set(X_test[col].unique()) ]
print(bad_cols)
print(len(bad_cols))
df.drop(columns=bad_cols, axis=1, inplace=True)
print(df.shape)
X_test.drop(columns=bad_cols, axis=1, inplace=True)
print(X_test.shape)
for col in df.columns:
    if df[col].isnull().sum()>0:
        print(col, df[col].isnull().sum())
for col in X_test.columns:
    if X_test[col].isnull().sum()>0:
        print(col, X_test[col].isnull().sum())
df.dropna(axis=1, how='all',thresh=700, inplace=True)
print(df.shape)
X_test.dropna(axis=1, how='all',thresh=700, inplace=True)
print(X_test.shape)
print(df.columns)
print(len(df.columns))
miss_obj=[]
miss_flt=[]

for col in df.columns:
    if df[col].isnull().sum()>0:
        if df[col].dtypes == 'object':
            miss_obj.append(df[col].name)
        elif df[col].dtypes == 'float64':
            miss_flt.append(df[col].name)     
        print(col,': missing =', df[col].isnull().sum())

print('miss_obj = ',miss_obj)
print('miss_flt = ', miss_flt)
for cols in miss_flt:
    df[cols].fillna(df[cols].mean(), inplace=True)
for cols in miss_obj:
    df[cols].fillna(str(df[cols].mode()), inplace=True)
obj_cols=[]

for col in df.columns:
    if df[col].dtypes == 'object':
        obj_cols.append(df[col].name)

print('obj_cols = ',obj_cols)
print(len(obj_cols))
data_cols = ['MSSubClass', 'LotFrontage', 'LotArea', 'Street', 'LotShape',
       'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
       'BldgType', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond',
       'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
       'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
       'HeatingQC', 'CentralAir', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
       'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish',
       'GarageCars', 'GarageArea', 'GarageCond', 'PavedDrive', 'WoodDeckSF',
       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       'MiscVal', 'MoSold', 'YrSold', 'SaleCondition', 'SalePrice']

import category_encoders as ce
count_enc = ce.CountEncoder()
count_encoded = count_enc.fit_transform(df[obj_cols])
df = df[data_cols].join(count_encoded.add_suffix("_count"))

df.shape
'GarageCond' in df.columns
df_new=df.select_dtypes(exclude=['object'])
df_new.shape
'GarageCond' in df.columns
df_new
feature_cols = df_new.columns.drop('SalePrice')
X=df_new[feature_cols]
X
y=df_new.SalePrice
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=0)
X_train
para = list(range(100, 202, 100))
print(para)
y_train
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, mean_absolute_error
results = {}
for n in para:
    print('para=', n)
    model = RandomForestClassifier(n_estimators=n)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
#     accu = accuracy_score(y_true=y_valid, y_pred=preds)
#     f1 = f1_score(y_true=y_valid, y_pred=preds, average='micro')
#     print(classification_report(y_true=y_valid, y_pred=preds))
    mae = mean_absolute_error(y_true=y_valid, y_pred=preds)
    print(mae)
    print('--------------------------')
    results[n] = mae
import matplotlib.pylab as plt

# sorted by key, return a list of tuples
lists = sorted(results.items()) 
p, a = zip(*lists) # unpack a list of pairs into two tuples
plt.plot(p, a)
plt.show()
best_para = max(results, key=results.get)
print('best para', best_para)
print('value', results[best_para])
X_test
print(X_test.columns)
print(len(X_test.columns))
miss_obj_X=[]
miss_flt_X=[]

for colX in X_test.columns:
    if X_test[colX].isnull().sum()>0:
        if X_test[colX].dtypes == 'object':
            miss_obj_X.append(X_test[colX].name)
        elif X_test[colX].dtypes == 'float64':
            miss_flt_X.append(X_test[colX].name)     
        print(colX,': missing =', X_test[colX].isnull().sum())
print('miss_obj_X = ',miss_obj_X)
print('miss_flt_X = ', miss_flt_X)
for cols in miss_flt_X:
    X_test[cols].fillna(df[cols].mean(), inplace=True)
for cols in miss_obj_X:
    X_test[cols].fillna(str(df[cols].mode()), inplace=True)
obj_colXs=[]

for colX in X_test.columns:
    if X_test[colX].dtypes == 'object':
        obj_colXs.append(X_test[colX].name)

print('obj_colXs = ',obj_colXs)
print(len(obj_colXs))
data_cols = ['MSSubClass', 'LotFrontage', 'LotArea', 'Street', 'LotShape',
       'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
       'BldgType', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond',
       'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
       'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
       'HeatingQC', 'CentralAir', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
       'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish',
       'GarageCars', 'GarageArea', 'GarageCond', 'PavedDrive', 'WoodDeckSF',
       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       'MiscVal', 'MoSold', 'YrSold', 'SaleCondition']

import category_encoders as ce
count_enc = ce.CountEncoder()
count_encoded = count_enc.transform(df[obj_cols])
X_test = X_test[data_cols].join(count_encoded.add_suffix("_count"))

X_test
for col in X_test.columns:
    if X_test[col].isnull().sum() > 0:
        print(col, X_test[col].isnull().sum())
X_test.columns
print('miss_obj_X = ',miss_obj_X)
print('miss_flt_X = ', miss_flt_X)
for colXs in miss_flt_X:
    X_test[colXs].fillna(df_new[colXs].mean(), inplace=True)
for colX in X_test.columns:
    if X_test[colX].isnull().sum()>0:
         print(colX,
              ': missing =', X_test[colX].isnull().sum(),
              ' type = ', X_test[colX].dtypes)
df['GarageCond'].unique()
miss_obj_X
for colX in X_test.columns:
    if X_test[colX].isnull().sum()>0:
        print(colX,
              ': missing =', X_test[colX].isnull().sum(),
              ' type = ', X_test[colX].dtypes)

X_test
obj_colXs=[]

for colX in X_test.columns:
    if X_test[colX].dtypes == 'object':
        obj_colXs.append(X_test[colX].name)

print('obj_colXs = ',obj_colXs)
X_test.isnull().sum()
miss_obj_X=[]
miss_flt_X=[]

for colX in X_test.columns:
    if X_test[colX].isnull().sum()>0:
        if X_test[colX].dtypes == 'object':
            miss_obj.append(X_test[colX].name)
        elif X_test[colX].dtypes == 'float64':
            miss_flt.append(X_test[colX].name)     
        print(colX,': missing =', X_test[colX].isnull().sum())

print('miss_obj = ',miss_obj)
print('miss_flt = ', miss_flt)
from sklearn.ensemble import RandomForestClassifier

final_model = RandomForestClassifier(n_estimators=best_para)
final_model.fit(X, y)

preds_test = final_model.predict(X_test)


output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
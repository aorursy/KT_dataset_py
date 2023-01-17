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
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

print(train['MSSubClass'].value_counts())
print(test['MSSubClass'].value_counts())
train.info()
def fill_nan(column, filler):
    return column.fillna(value=filler)

cols_to_fill = [['Alley', 'None'],
                ['MasVnrType', 'None'],
                ['MasVnrArea', 0.0],
                ['BsmtQual', 'NA'],
                ['BsmtCond', 'NA'],
                ['BsmtExposure','NA'],
                ['BsmtFinType1', 'NA'],
                ['BsmtFinType2', 'NA'],
                ['FireplaceQu', 'NA'],
                ['LotFrontage', 0.0],
                ['GarageType', 'NA'],
                ['GarageFinish', 'NA'],
                ['GarageQual', 'NA'],
                ['GarageCond', 'NA'],
                ['Fence', 'NA']
               ]

for col in cols_to_fill:
    train[col[0]]=fill_nan(train[col[0]],col[1])
    test[col[0]]=fill_nan(test[col[0]],col[1])

print(train.shape)
print(test.shape)
train.info()
train['HasPool']=train['PoolQC'].notna().astype(int)
train.drop(columns=['PoolQC','PoolArea'], inplace=True)

test['HasPool']=test['PoolQC'].notna().astype(int)
test.drop(columns=['PoolQC','PoolArea'], inplace=True)
train.drop(columns='MiscFeature', inplace=True)

test.drop(columns='MiscFeature', inplace=True)
train.dropna(subset=['Electrical'], inplace=True)

#test.dropna(subset=['Electrical'], inplace=True)
train.drop(columns='Id', inplace=True)
testId = test['Id']
test.drop(columns='Id', inplace=True)
print(train.shape)
print(test.shape)
train['YrsSinceConst'] = train['YrSold'] - train['YearRemodAdd']
train.drop(columns=['MoSold', 'YrSold', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt'], inplace=True)

test['YrsSinceConst'] = test['YrSold'] - test['YearRemodAdd']
test.drop(columns=['MoSold', 'YrSold', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt'], inplace=True)

basement_drops=['BsmtExposure', 'BsmtFinType1','BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF']
train.drop(columns=basement_drops, inplace=True)

test.drop(columns=basement_drops, inplace=True)
print(train.shape)
print(test.shape)
garage_drops=['GarageType','GarageFinish', 'GarageCars']
train.drop(columns=garage_drops, inplace=True)

test.drop(columns=garage_drops, inplace=True)
def to_numeric(row):
    if row == "Ex":
        return 5
    elif row=="Gd":
        return 4
    elif row=="TA":
        return 3
    elif row=="Fa":
        return 2
    elif row=="Po":
        return 1
    else:
        return 0
    
ordinal_cols = ['ExterQual', 'GarageCond', 'GarageQual', 'HeatingQC', 'FireplaceQu', 'KitchenQual', 'BsmtQual', 'BsmtCond', 'ExterCond']

for col in ordinal_cols:
    train[col]=train[col].apply(to_numeric)
    test[col]=test[col].apply(to_numeric)
print(train.shape)
print(test.shape)
def shape(row):
    if row=='Reg':
        return 4
    elif row=='IR1':
        return 3
    elif row=='IR2':
        return 2
    elif row=='IR3':
        return 1
    
train['LotShape'] = train['LotShape'].apply(shape)
test['LotShape'] = test['LotShape'].apply(shape)
def slope(row):
    if row=='Gtl':
        return 3
    elif row=='Mod':
        return 2
    elif row=='Sev':
        return 1
    
train['LandSlope'] = train['LandSlope'].apply(slope)
test['LandSlope'] = test['LandSlope'].apply(slope)
def functional(row):
    if row=="Sal":
        return 0
    elif row =="Sev":
        return 1
    elif row =="Maj2":
        return 2
    elif row =="Maj1":
        return 3
    elif row =="Mod":
        return 4
    elif row =="Min2":
        return 5
    elif row =="Min1":
        return 6
    elif row =="Typ":
        return 7
    
train['Functional'] = train['Functional'].apply(functional)
test['Functional'] = test['Functional'].apply(functional)
print(train.shape)
print(test.shape)
test["MSZoning"].replace(np.NaN,"RL", inplace=True)
test["Utilities"].replace(np.NaN,"AllPub", inplace=True)
test["Utilities"].replace(np.NaN,"AllPub", inplace=True)
test["TotalBsmtSF"].replace(np.NAN, test["TotalBsmtSF"].mean(), inplace=True)
test["BsmtHalfBath"].replace(np.NaN,"0.0", inplace=True)
test["BsmtFullBath"].replace(np.NAN, test["BsmtFullBath"].mean(), inplace=True)
test["Functional"].replace(np.NaN,7.0, inplace=True)
test["SaleType"].replace(np.NaN, 'WD', inplace=True)
test["GarageArea"].replace(np.NAN, test["GarageArea"].mean(), inplace=True)
test.isna().sum()
train['Set']='train'
test['Set']='test'

data=pd.concat([train,test])
# train['Conditions'] = train['Condition1'] + ',' + train['Condition2']
# train['Exteriors'] = train["Exterior1st"] + ',' + train["Exterior2nd"]

# train.drop(columns=['Condition1', 'Condition2', 'Exterior1st', 'Exterior2nd'], inplace=True)

# condition_dummies = train['Conditions'].str.get_dummies(sep=',')
# exterior_dummies = train['Exteriors'].str.get_dummies(sep=',')

# train = pd.concat([train, condition_dummies, exterior_dummies], axis=1)

# train.drop(columns=['Conditions', 'Exteriors'], inplace=True)




# test['Conditions'] = test['Condition1'] + ',' + test['Condition2']
# test['Exteriors'] = test["Exterior1st"] + ',' + test["Exterior2nd"]

# test.drop(columns=['Condition1', 'Condition2', 'Exterior1st', 'Exterior2nd'], inplace=True)

# condition_dummies = test['Conditions'].str.get_dummies(sep=',')
# exterior_dummies = test['Exteriors'].str.get_dummies(sep=',')

# test = pd.concat([test, condition_dummies, exterior_dummies], axis=1)

# test.drop(columns=['Conditions', 'Exteriors'], inplace=True)




data['Conditions'] = data['Condition1'] + ',' + data['Condition2']
data['Exteriors'] = data["Exterior1st"] + ',' + data["Exterior2nd"]

data.drop(columns=['Condition1', 'Condition2', 'Exterior1st', 'Exterior2nd'], inplace=True)

condition_dummies = data['Conditions'].str.get_dummies(sep=',')
exterior_dummies = data['Exteriors'].str.get_dummies(sep=',')

data = pd.concat([data, condition_dummies, exterior_dummies], axis=1)

data.drop(columns=['Conditions', 'Exteriors'], inplace=True)



print(train.shape)
print(test.shape)
print(data.shape)
dummies = ['MSSubClass', 'LandContour','MSZoning', 'Street', 'Alley', 'Utilities', 'LotConfig', 'Neighborhood', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'Electrical','PavedDrive', 'Fence', 'SaleType', 'SaleCondition']    

train = pd.get_dummies(train, columns=dummies, drop_first=True)
test = pd.get_dummies(test, columns=dummies, drop_first=True)

data = pd.get_dummies(data, columns=dummies, drop_first=True)
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib notebook
#sns.set(style="white")

# Compute the correlation matrix
#corr = train.corr()

# Generate a mask for the upper triangle
#mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure
#f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
#cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
#sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
train.info(verbose=True)
pd.set_option('display.max_rows', None)
np.absolute(train.corr()['SalePrice']).sort_values(ascending=False)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

train = data[data['Set']=='train']
test = data[data['Set']=='test']

train.drop(columns='Set', inplace=True)
test.drop(columns=['Set','SalePrice'], inplace=True)

print(train.shape)
print(test.shape)

y = train['SalePrice']
X = train.drop(columns="SalePrice")

y=np.log(y)

#X = X.drop(columns=missing_cols,inplace=True)

lr = LinearRegression()
lr.fit(X, y)
# predictions = lr.predict(X)

# rmse = mean_squared_error(y, predictions)**.5
# print(rmse)
# print(model.score(X,y))


# from sklearn.model_selection import cross_val_score
# print(cross_val_score(lr, X, y, cv=5, scoring='neg_root_mean_squared_error'))


# from sklearn.linear_model import Ridge

# for i in range (-4, 5):
#     alpha = 10**i
#     rm = Ridge(alpha=alpha)
#     print("alpha: ", alpha ,sum(cross_val_score(rm, X, y, cv=10, scoring='neg_root_mean_squared_error'))/10)
# Get missing columns in the training test
# missing_cols = set( train.columns ) - set( test.columns )
# # Add a missing column in test set with default value equal to 0
# for c in missing_cols:
#     test[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
#test = test[train.columns]

# lr = LinearRegression()
# lr.fit(X, y)

# test.drop(columns="SalePrice", inplace=True)
predictions = lr.predict(test)
predictions = np.exp(predictions)
print(pd.Series(predictions))
submit = pd.concat([testId, pd.Series(predictions)], axis=1)
submit.head(10)
submit.rename(columns={0:"SalePrice"}, inplace=True)
submit.to_csv('submission.csv', index=False)
submit.head()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from scipy.stats import norm
from scipy import stats
from sklearn import preprocessing
train_data = pd.read_csv('../input/train.csv') #train_data
test_data = pd.read_csv('../input/test.csv')  #test_data
train_data.head(5)
test_data.head(5)
#Numerical features NaN value filling
#train_data
num_feat_index = train_data.dtypes[train_data.dtypes != "object"].index
print('Train Data Numerical features are:' )
print(num_feat_index)
train_num_na = train_data[num_feat_index].isnull().sum()/len(train_data)
train_num_na=train_num_na.drop(train_num_na[train_num_na == 0].index).sort_values(ascending = False)
train_num_na = pd.DataFrame({'Train Data Missing Ratio': train_num_na})
train_num_na
plt.xticks(rotation = '0')
sns.barplot(x=train_num_na.index, y=train_num_na.iloc[:,0])
#for column in ['LotFrontage','GarageYrBlt']:
 #   train_data[column]=train_data[column].fillna(train_data[column].mode()[0])
train_data['LotFrontage']=train_data['LotFrontage'].fillna(train_data['LotFrontage'].mode()[0])
#Text type features NaN value filling
#train_data
text_feat_index = train_data.dtypes[train_data.dtypes == "object"].index
print('Train Data Text Type features are:' )
print(text_feat_index)
train_text_na = train_data[text_feat_index].isnull().sum()/len(train_data)
train_text_na=train_text_na.drop(train_text_na[train_text_na == 0].index).sort_values(ascending = False)
train_text_na = pd.DataFrame({'Train Data Missing Ratio': train_text_na})
plt.xticks(rotation = '90')
sns.barplot(x = train_text_na.index, y = train_text_na.iloc[:,0])
train_text_na
train_data.loc[train_data['PoolArea']>0, ['PoolQC','PoolArea']]
train_data['PoolQC']=train_data['PoolQC'].fillna('None')
columns = ['MiscFeature','Alley','Fence','FireplaceQu','Electrical']
for column in columns:
    train_data[column] = train_data[column].fillna('None')
train_data.loc[train_data['GarageCars']==0, ['GarageArea','GarageCond','GarageQual','GarageFinish','GarageType','GarageCars','GarageYrBlt']].head(10)
train_data.loc[train_data['GarageCars']==0, ['GarageYrBlt']] = train_data.loc[train_data['GarageCars']==0, ['GarageYrBlt']].fillna(0)
train_data.loc[train_data['GarageCars']==0, ['GarageCond','GarageQual','GarageFinish','GarageType']]=train_data.loc[train_data['GarageCars']==0, ['GarageCond','GarageQual','GarageFinish','GarageType']].fillna('None')
inx = (train_data['GarageCars']>0)&((train_data['GarageCond'].isnull())|(train_data['GarageQual'].isnull())|(train_data['GarageFinish'].isnull())|(train_data['GarageType'].isnull())|(train_data['GarageArea'].isnull())|(train_data['GarageYrBlt'].isnull()))
train_data.loc[inx, ['GarageArea','GarageCars','GarageYrBlt','GarageCond','GarageQual','GarageFinish','GarageType']]
train_data[['GarageArea','GarageCars','GarageYrBlt','GarageCond','GarageQual','GarageFinish','GarageType']].isnull().sum()
train_data.loc[train_data['TotalBsmtSF']==0,['TotalBsmtSF','BsmtFinType1','BsmtFinType2','BsmtExposure','BsmtCond','BsmtQual']]
for column in ['BsmtFinType1','BsmtFinType2','BsmtExposure','BsmtCond','BsmtQual']:
    train_data[column]=train_data[column].fillna('None')
inx = train_data['MasVnrArea'].isnull() | train_data['MasVnrType'].isnull()
train_data.loc[inx,['MasVnrArea','MasVnrType']]
train_data['MasVnrType']=train_data['MasVnrType'].fillna('None')
train_data['MasVnrArea'] = train_data['MasVnrArea'].fillna(0)
print(len(text_feat_index[train_data[text_feat_index].isnull().sum()/len(train_data)==0])/len(text_feat_index)*100,'%')
print(len(train_data.columns[train_data.isnull().sum()==0])/len(train_data.columns)*100,'%')
# Function collecting dummies variables
def dummies(data, columns):
    dummiesgroup = {}
    for column in columns:
        dummies = {}
        variables = train_data[column].unique()
        num = 1
        for variable in variables:
            if variable == 'None':
                dummies[variable] = 0
            else:
                dummies[variable] = num
            num += 1
        dummiesgroup[column] = dummies
    return(dummiesgroup)   
dummiesgroup = dummies(train_data, text_feat_index)
dummiesgroup
new_train_data = train_data.copy()
for column in text_feat_index:
    new_train_data[column] = train_data[column].map(dummiesgroup[column])
test_data.head(10)
#test_data
num_feat_index = test_data.dtypes[test_data.dtypes != "object"].index
print('Test Data Numerical features are:' )
print(num_feat_index)
test_num_na = test_data[num_feat_index].isnull().sum()/len(test_data)
test_num_na=test_num_na.drop(test_num_na[test_num_na == 0].index).sort_values(ascending = False)
test_num_na = pd.DataFrame({'Test Data Missing Ratio': test_num_na})
test_num_na
#for column in ['LotFrontage','GarageYrBlt']:
 #   test_data[column]=test_data[column].fillna(train_data[column].mode()[0])
test_data['LotFrontage']=test_data['LotFrontage'].fillna(train_data['LotFrontage'].mode()[0])   
txt_feat_index = test_data.dtypes[test_data.dtypes == "object"].index
print('Test Data text type features are:' )
print(txt_feat_index)
test_txt_na = test_data[txt_feat_index].isnull().sum()/len(test_data)
test_txt_na=test_txt_na.drop(test_txt_na[test_txt_na == 0].index).sort_values(ascending = False)
test_txt_na = pd.DataFrame({'Test Data Missing Ratio': test_txt_na})
test_txt_na
test_data.loc[test_data['PoolArea']>0, ['PoolQC','PoolArea']]
test_data.loc[test_data['PoolArea']>0,'PoolQC'] = test_data.loc[test_data['PoolArea']>0,'PoolQC'].fillna(train_data.loc[train_data['PoolArea']>0,'PoolQC'].mode()[0])
test_data.loc[test_data['PoolArea']==0,'PoolQC'] = test_data.loc[test_data['PoolArea']==0,'PoolQC'].fillna('None')
columns = ['MiscFeature','Alley','Fence','FireplaceQu']
for column in columns:
    test_data[column] = test_data[column].fillna('None')
test_data.loc[test_data['GarageArea']==0, ['GarageArea','GarageCars','GarageYrBlt','GarageCond','GarageQual','GarageFinish','GarageType']].head(5)
test_data.loc[test_data['GarageArea']==0, ['GarageArea','GarageCond','GarageQual','GarageFinish','GarageType']] = test_data.loc[test_data['GarageArea']==0, ['GarageArea','GarageCond','GarageQual','GarageFinish','GarageType']].fillna('None')
test_data.loc[test_data['GarageArea']==0,'GarageYrBlt'] = test_data.loc[test_data['GarageArea']==0,'GarageYrBlt'].fillna(0)
inx = (test_data['GarageArea'] > 0) & (test_data['GarageCond'].isnull() | test_data['GarageQual'].isnull() | test_data['GarageFinish'].isnull() | test_data['GarageType'].isnull() | (test_data['GarageYrBlt'].isnull()) | (test_data['GarageCars'].isnull()))
test_data.loc[inx, ['GarageArea','GarageCond','GarageQual','GarageFinish','GarageType','GarageCars','GarageYrBlt']]
test_data.loc[inx,'GarageCond'] = test_data.loc[inx,'GarageCond'].fillna(train_data.loc[train_data['GarageArea']>0,'GarageCond'].mode()[0])
test_data.loc[inx, 'GarageQual'] = test_data.loc[inx,'GarageQual'].fillna(train_data.loc[train_data['GarageArea']>0,'GarageQual'].mode()[0])
test_data.loc[inx,'GarageFinish'] = test_data.loc[inx,'GarageFinish'].fillna(train_data.loc[train_data['GarageType'] == 'Detchd','GarageFinish'].mode()[0])
test_data.loc[inx,'GarageYrBlt'] = test_data.loc[inx, 'GarageYrBlt'].fillna(train_data.loc[train_data['GarageArea'] > 0,'GarageYrBlt'].mode()[0])
test_data[['GarageArea','GarageCond','GarageQual','GarageFinish','GarageType']].isnull().sum()
inx = (test_data['GarageCond'].isnull() | test_data['GarageQual'].isnull() | test_data['GarageFinish'].isnull() | test_data['GarageType'].isnull())
test_data.loc[inx, ['GarageArea','GarageCond','GarageQual','GarageFinish','GarageType','GarageCars','GarageYrBlt']]
test_data.loc[inx,'GarageArea'] = test_data.loc[inx,'GarageArea'].fillna(train_data.loc[train_data['GarageType'] == 'Detchd','GarageArea'].mode()[0])
test_data.loc[inx,'GarageCars'] = test_data.loc[inx,'GarageCars'].fillna(train_data.loc[train_data['GarageType'] == 'Detchd','GarageCars'].mode()[0])
test_data.loc[inx,'GarageCond'] = test_data.loc[inx, 'GarageCond'].fillna(train_data.loc[train_data['GarageArea'] > 0,'GarageCond'].mode()[0])
test_data.loc[inx,'GarageQual'] = test_data.loc[inx, 'GarageQual'].fillna(train_data.loc[train_data['GarageArea'] > 0,'GarageQual'].mode()[0])
test_data.loc[inx,'GarageFinish'] = test_data.loc[inx, 'GarageFinish'].fillna(train_data.loc[train_data['GarageArea'] > 0,'GarageFinish'].mode()[0])
test_data.loc[inx,'GarageYrBlt'] = test_data.loc[inx, 'GarageYrBlt'].fillna(train_data.loc[train_data['GarageArea'] > 0,'GarageYrBlt'].mode()[0])
test_data[['GarageArea','GarageCars','GarageYrBlt','GarageCond','GarageQual','GarageFinish','GarageType']].isnull().sum()
inx = test_data['MasVnrType'].isnull() | test_data['MasVnrArea'].isnull()
test_data.loc[inx, ['MasVnrType','MasVnrArea']]
test_data.loc[(test_data['MasVnrType'].isnull()) & (test_data['MasVnrArea']>0),'MasVnrType'] = test_data.loc[(test_data['MasVnrType'].isnull()) & (test_data['MasVnrArea']>0),'MasVnrType'].fillna(train_data.loc[train_data['MasVnrArea']>0,'MasVnrType'].mode()[0])
test_data.loc[inx,'MasVnrType'] = test_data.loc[inx,'MasVnrType'].fillna('None')
test_data.loc[inx,'MasVnrArea'] = test_data.loc[inx,'MasVnrArea'].fillna(0)
test_data[['MasVnrType','MasVnrArea']].isnull().sum()
inx = test_data['BsmtQual'].isnull()|test_data['BsmtCond'].isnull()|test_data['BsmtExposure'].isnull()|test_data['BsmtFinType1'].isnull()\
|test_data['BsmtFinType2'].isnull()|test_data['BsmtFinSF1'].isnull()|test_data['BsmtFinSF2'].isnull()|test_data['TotalBsmtSF'].isnull()\
|test_data['BsmtUnfSF'].isnull()|test_data['BsmtFullBath'].isnull()|test_data['BsmtHalfBath'].isnull()
test_data.loc[inx,['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2','TotalBsmtSF','BsmtUnfSF','BsmtFullBath','BsmtHalfBath']]
inx1 = (test_data['BsmtQual'].isnull()|test_data['BsmtCond'].isnull()|test_data['BsmtExposure'].isnull()|test_data['BsmtFinType1'].isnull()\
|test_data['BsmtFinType2'].isnull())&((test_data['BsmtFinSF1'] == 0)&(test_data['BsmtFinSF2']==0)&(test_data['TotalBsmtSF']==0)\
&(test_data['BsmtUnfSF']==0)&(test_data['BsmtFullBath']==0)&(test_data['BsmtHalfBath']==0))
test_data.loc[inx1, ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2','TotalBsmtSF','BsmtUnfSF','BsmtFullBath','BsmtHalfBath']]
test_data.loc[inx1, ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']]=test_data.loc[inx1, ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']].fillna('None')
inx = test_data['BsmtQual'].isnull()|test_data['BsmtCond'].isnull()|test_data['BsmtExposure'].isnull()|test_data['BsmtFinType1'].isnull()\
|test_data['BsmtFinType2'].isnull()|test_data['BsmtFinSF1'].isnull()|test_data['BsmtFinSF2'].isnull()|test_data['TotalBsmtSF'].isnull()\
|test_data['BsmtUnfSF'].isnull()|test_data['BsmtFullBath'].isnull()|test_data['BsmtHalfBath'].isnull()
test_data.loc[inx,['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2','TotalBsmtSF','BsmtUnfSF','BsmtFullBath','BsmtHalfBath']]
test_data.loc[inx,['BsmtFinSF1','BsmtFinSF2','TotalBsmtSF','BsmtUnfSF','BsmtFullBath','BsmtHalfBath']] = test_data.loc[inx,['BsmtFinSF1','BsmtFinSF2','TotalBsmtSF','BsmtUnfSF','BsmtFullBath','BsmtHalfBath']].fillna(0)
test_data.loc[inx,['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2','TotalBsmtSF','BsmtUnfSF','BsmtFullBath','BsmtHalfBath']]
test_data.loc[test_data['TotalBsmtSF']==0,['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']] = test_data.loc[test_data['TotalBsmtSF']==0,['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']].fillna('None')
test_data.loc[inx,'BsmtExposure'] = test_data.loc[inx,'BsmtExposure'].fillna('No')
test_data.loc[inx, 'BsmtCond'] = test_data.loc[inx, 'BsmtCond'].fillna(train_data.loc[train_data['TotalBsmtSF']>0, 'BsmtCond'].mode()[0])
test_data.loc[inx, 'BsmtQual'] = test_data.loc[inx, 'BsmtQual'].fillna(train_data.loc[train_data['TotalBsmtSF']>0, 'BsmtQual'].mode()[0])
inx1 = (test_data['BsmtQual'].isnull()|test_data['BsmtCond'].isnull()|test_data['BsmtExposure'].isnull()|test_data['BsmtFinType1'].isnull()\
|test_data['BsmtFinType2'].isnull())&((test_data['BsmtFinSF1'] == 0)&(test_data['BsmtFinSF2']==0)&(test_data['TotalBsmtSF']==0)\
&(test_data['BsmtUnfSF']==0)&(test_data['BsmtFullBath']==0)&(test_data['BsmtHalfBath']==0))
test_data.loc[inx1, ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2','TotalBsmtSF','BsmtUnfSF','BsmtFullBath','BsmtHalfBath']]
inx = test_data['MSZoning'].isnull() | test_data['Utilities'].isnull() | test_data['Functional'].isnull()
test_data.loc[inx,['MSZoning', 'Utilities', 'Functional']]
for column in ['MSZoning', 'Utilities', 'Functional']:
    test_data.loc[inx,column] = test_data.loc[inx,column].fillna(train_data[column].mode()[0])
inx = test_data['KitchenQual'].isnull()
test_data.loc[inx,'KitchenQual']
test_data.loc[inx,'KitchenQual'] = test_data.loc[inx,'KitchenQual'].fillna(train_data['KitchenQual'].mode()[0])
inx = test_data['SaleType'].isnull()
test_data.loc[inx,'SaleType']
test_data.loc[inx, 'SaleType'] = test_data.loc[inx, 'SaleType'].fillna(train_data['SaleType'].mode()[0])
inx = test_data['Exterior1st'].isnull() | test_data['Exterior2nd'].isnull()
test_data.loc[inx,['Exterior1st','Exterior2nd']]
test_data.loc[inx,['Exterior1st']] = test_data.loc[inx,['Exterior1st']].fillna(train_data['Exterior1st'].mode()[0])
test_data.loc[inx,['Exterior2nd']] = test_data.loc[inx, ['Exterior2nd']].fillna(train_data['Exterior2nd'].mode()[0])
print(len(test_data.columns[test_data.isnull().sum()==0])/len(test_data.columns)*100,'%')
new_test_data = test_data.copy()
for column in text_feat_index:
    new_test_data[column] = test_data[column].map(dummiesgroup[column])
new_test_data.head(5)
train_x = new_train_data[['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',\
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',\
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',\
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',\
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',\
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',\
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',\
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',\
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',\
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',\
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',\
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',\
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',\
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',\
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',\
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',\
       'SaleCondition']]
test_x = new_test_data[['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',\
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',\
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',\
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',\
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',\
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',\
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',\
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',\
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',\
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',\
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',\
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',\
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',\
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',\
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',\
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',\
       'SaleCondition']]
scaler = preprocessing.MinMaxScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)
test_x
train_x = pd.DataFrame(train_x, columns = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',\
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',\
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',\
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',\
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',\
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',\
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',\
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',\
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',\
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',\
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',\
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',\
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',\
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',\
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',\
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',\
       'SaleCondition'])
test_x = pd.DataFrame(test_x, columns = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',\
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',\
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',\
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',\
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',\
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',\
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',\
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',\
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',\
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',\
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',\
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',\
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',\
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',\
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',\
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',\
       'SaleCondition'])
train_x.head(5)
train_x.shape
test_x.head(5)
sns.distplot(train_data['SalePrice'],fit=norm)
mu, sigma = norm.fit(train_data['SalePrice'])
plt.legend(['Normal dist.\n($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(mu,sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice Distribution')
plt.figure()
stats.probplot(train_data['SalePrice'],plot=plt)
train_y = np.log1p(train_data['SalePrice'])
sns.distplot(train_y,fit=norm)
mu,sigma = norm.fit(train_y)
plt.legend(['Normal dist.\n($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(mu, sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice Distribution')
plt.figure()
stats.probplot(train_data['SalePrice'],plot=plt)
corr = new_train_data.iloc[:,1::].corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corr, vmax = 0.9, square=True)
#rank features according to the correlation coefficient with SalePrice 
columns = np.abs(corr['SalePrice']).sort_values(ascending = False).index[1::].tolist()
print(columns)
#Looking for high correlated variables
n=0
columnsA=[]
columnsB=[]
while n < len(columns):
    if columns[n] not in columnsB:
        cols = corr.columns[corr[columns[n]]>=0.8].tolist() #Assume corr coefficient 0.8 as valve value.
        cols.remove(columns[n])
        if len(cols)>0:
            for col in cols:
                if col!='SalePrice':
                    columnsB.append(col)
            print(n, columns[n], cols)
        columnsA.append(columns[n])
    n+=1
print(columnsA)
print(columnsB)
from sklearn.model_selection import KFold, cross_val_score, train_test_split
n_folds = 5
def rmse_kfold(model):
    kf = KFold(n_folds, shuffle = True, random_state = 42).get_n_splits(train_x[columnsA].values)
    rmse = np.sqrt(-cross_val_score(model, train_x[columnsA].values, train_y, scoring = "neg_mean_squared_error", cv = kf ))
    return(rmse)
from sklearn import linear_model
lr = linear_model.LinearRegression()
lr_loss = rmse_kfold(lr)
from sklearn.svm import SVR
#linear kernel
linear_svr_loss = []
for i in [1,10,1e2]:
#for i in [1,10,1e2,1e3,1e4]:
    linear_svr = SVR(kernel = 'linear', C=i)
    linear_svr_loss.append(rmse_kfold(linear_svr).mean())
plt.plot([1,10,1e2],linear_svr_loss)
plt.xlabel('C')
plt.ylabel('mean-loss')
linear_svr = SVR(kernel = 'linear', C=1)
linear_svr_loss = rmse_kfold(linear_svr)
#poly kernel
for i in [1,10,1e2,1e3,1e4]:
    poly_svr_loss=[]
    for j in np.linspace(2,9,10):
        poly_svr = SVR(kernel = 'poly',C=i, degree=j)
        poly_svr_loss.append(rmse_kfold(poly_svr).mean())
    plt.plot(np.linspace(2,9,10), poly_svr_loss, label='C='+str(i))
    plt.legend()
plt.xlabel('degree')
plt.ylabel('mean-loss')
poly_svr = SVR(kernel = 'poly',C=100, degree=2)
poly_svr_loss=rmse_kfold(poly_svr)
#rbf kernel
for i in [1,10,1e2,1e3,1e4]:
    rbf_svr_loss = []
    for j in np.linspace(0.1,1,10):
        rbf_svr = SVR(kernel = 'rbf', C=i, gamma=j)
        rbf_svr_loss.append(rmse_kfold(rbf_svr).mean())
    plt.plot(np.linspace(0.1,1,10), rbf_svr_loss, label='C='+str(i))
    plt.legend()
plt.xlabel('gamma')
plt.ylabel('mean-loss')
rbf_svr = SVR(kernel = 'rbf',C=1,gamma=0.1)
rbf_svr_loss = rmse_kfold(rbf_svr)
from sklearn.neighbors import KNeighborsRegressor
knn_loss = []
for n_neighbors in range(1,21):
    knn = KNeighborsRegressor(n_neighbors, weights = 'uniform' )
    knn_loss.append(rmse_kfold(knn).mean())
plt.plot(np.linspace(1,20,20), knn_loss)
plt.xlabel('n-neighbors')
plt.ylabel('mean-loss')
knn = KNeighborsRegressor(6, weights = 'uniform' )
knn_loss = rmse_kfold(knn)
from sklearn.tree import DecisionTreeRegressor
dtr_loss=[]
for n in range(1,11):
    dtr = DecisionTreeRegressor(max_depth = n)
    dtr_loss.append(rmse_kfold(dtr).mean())
plt.plot(np.linspace(1,10,10), dtr_loss)
plt.xlabel('max_depth')
plt.ylabel('mean-loss')
dtr = DecisionTreeRegressor(max_depth = 7)
dtr_loss=rmse_kfold(dtr)
from sklearn.kernel_ridge import KernelRidge    
#linear kernel
kr_linear_loss = []
for i in np.linspace(0.1,2.0,8):
    kr = KernelRidge(alpha = i,kernel = 'linear')
    kr_linear_loss.append(rmse_kfold(kr).mean())
plt.plot(np.linspace(0.1,2.0,8), kr_linear_loss)
plt.xlabel('alpha')
plt.ylabel('mean-loss')
kr_linear = KernelRidge(alpha = 1.3,kernel = 'linear')
kr_linear_loss = rmse_kfold(kr_linear)
#poly kernel
for j in np.linspace(0.1,2.0,8):
    kr_poly_loss = []
    for i in np.linspace(0.01,0.1,10):
        kr = KernelRidge(alpha = j,kernel = 'poly', gamma = i)
        kr_poly_loss.append(rmse_kfold(kr).mean())
    plt.plot(np.linspace(0.01,0.1,10), kr_poly_loss,label='alpha:{:.2f}'.format(j))
    plt.legend(loc='upper right')
plt.xlabel('gamma')
plt.ylabel('mean-loss')
kr_poly = KernelRidge(alpha = 0.1,kernel = 'poly',gamma = 0.05)
kr_poly_loss = rmse_kfold(kr_poly)
#rbf kernel
for j in np.linspace(0.00001,0.0001,10):
    kr_rbf_loss = []
    for i in np.linspace(0.0005,0.02,10):
        kr = KernelRidge(alpha = j,kernel = 'rbf', gamma = i)
        kr_rbf_loss.append(rmse_kfold(kr).mean())
    plt.plot(np.linspace(0.0005,0.02,10), kr_rbf_loss,label='alpha:{:.5f}'.format(j))
    plt.legend(loc='upper right')
plt.xlabel('gamma')
plt.ylabel('mean-loss')
kr_rbf = KernelRidge(alpha = 0.0001,kernel = 'rbf', gamma = 0.0025)
kr_rbf_loss=rmse_kfold(kr_rbf)
evaluating = {
    'lr': lr_loss,
    'linear_svr':linear_svr_loss,
    'polyl_svr':poly_svr_loss,
    'rbf_svr':rbf_svr_loss,
    'knn':knn_loss,
    'drt':dtr_loss,
    'kr_linear':kr_linear_loss,
    'kr_poly':kr_poly_loss,
    'kr_rbf':kr_rbf_loss
}
evaluating = pd.DataFrame(evaluating)
print(evaluating)
evaluating.plot.hist()
evaluating.hist(color='k',alpha=0.6,figsize=(8,7))
evaluating.describe()
n_folds = 5
def rmse_kfold1(model,corr_valve):
    corr1 = pd.DataFrame(corr.loc[columnsA,'SalePrice'])
    columns_split = corr1.loc[list(abs(corr1.loc[columnsA,'SalePrice'])>=corr_valve),'SalePrice'].index.tolist()
    kf = KFold(n_folds, shuffle = True, random_state = 42).get_n_splits(train_x[columns_split].values)
    rmse = np.sqrt(-cross_val_score(model, train_x[columns_split].values, train_y, scoring = "neg_mean_squared_error", cv = kf ))
    return(rmse)
loss={}
loss_mean={}
for valve in [0.5, 0.3, 0]:
    model_loss=[]
    model_mean_loss=[]
    for model in [lr,linear_svr,poly_svr,rbf_svr,knn,dtr,kr_poly,kr_rbf]:
        model_loss.append(rmse_kfold1(model,valve))
        model_mean_loss.append(rmse_kfold1(model,valve).mean())
    loss[str(valve)]=model_loss
    loss_mean[str(valve)]=model_mean_loss
sns.violinplot(data = pd.DataFrame(loss_mean))
plt.xlabel('valve value')
plt.ylabel('mean-loss')
loss_mean = pd.DataFrame(loss_mean,index = ['lr','linear_svr','poly_svr','rbf_svr','knn','dtr','kr_poly','kr_rbf'])
test_y_predict = np.expm1(rbf_svr.fit(train_x[columnsA].values,train_y).predict(test_x[columnsA].values))
submission = pd.DataFrame()
submission['Id'] = test_data['Id']
submission['SalePrice'] = test_y_predict
submission.to_csv('submission.csv',index = False)

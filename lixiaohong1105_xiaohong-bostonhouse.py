!pip install pyitlib
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost

from sklearn.model_selection import train_test_split

from pyitlib import discrete_random_variable as drv
train_raw = pd.read_csv('../input/train.csv')

train_raw.head()
train_raw.Alley.fillna('NoAlley', inplace=True)

train_raw.Fence.fillna('NoFence', inplace=True)

train_raw.FireplaceQu.fillna('NoFireplace', inplace=True)
# Impute Garage Related missing data, all Garage related fields are missing for the same records, hence refill with NoGarage

print(train_raw[train_raw.GarageType.isna() & train_raw.GarageCond.isna() & train_raw.GarageFinish.isna() & train_raw.GarageQual.isna()].shape)

print(train_raw[train_raw.GarageType.isna() | train_raw.GarageCond.isna() | train_raw.GarageFinish.isna() | train_raw.GarageQual.isna()].shape)

train_raw.GarageType.fillna('NoGarage', inplace=True)

train_raw.GarageCond.fillna('NoGarage', inplace=True)

train_raw.GarageFinish.fillna('NoGarage', inplace=True)

train_raw.GarageQual.fillna('NoGarage', inplace=True)
# Impute Basement Related missing data

print(train_raw[train_raw.BsmtExposure.isna() & train_raw.BsmtFinType2.isna()].shape)

print(train_raw[train_raw.BsmtExposure.isna() | train_raw.BsmtFinType2.isna()].shape)

print(train_raw[train_raw.BsmtFinType1.isna() & train_raw.BsmtCond.isna() & train_raw.BsmtQual.isna()].shape)

print(train_raw[train_raw.BsmtFinType1.isna() | train_raw.BsmtCond.isna() | train_raw.BsmtQual.isna()].shape)

print(train_raw[train_raw.BsmtFinType1.notna() & (train_raw.BsmtFinType2.isna()|train_raw.BsmtExposure.isna())][['BsmtExposure','BsmtFinType2','BsmtFinType1','BsmtCond','BsmtQual']])

# Most Basement related fields are missing for the same records, hence refill with NoBasement

train_raw.at[332, 'BsmtFinType2'] = 'GLQ'

train_raw.at[948, 'BsmtExposure'] = 'No'

train_raw.BsmtExposure.fillna('NoBasement', inplace=True)

train_raw.BsmtFinType2.fillna('NoBasement', inplace=True)

train_raw.BsmtFinType1.fillna('NoBasement', inplace=True)

train_raw.BsmtCond.fillna('NoBasement', inplace=True)

train_raw.BsmtQual.fillna('NoBasement', inplace=True)
# Impute PoolQC

print(train_raw[train_raw.PoolArea == 0].shape)

print(train_raw[train_raw.PoolQC.isna()].shape)

train_raw.PoolQC.fillna('NoPool', inplace=True)
# Impute Masonry Veneer, find the most correlated field to Masonry Veneer, then impute based on the correlated fields

print(train_raw[train_raw.MasVnrArea.isna() & train_raw.MasVnrType.isna()].shape)

print(train_raw[train_raw.MasVnrArea.isna() | train_raw.MasVnrType.isna()].shape)

print(train_raw.groupby(['MasVnrType']).agg({'SalePrice':[np.mean, np.size]}))

# Fill Masonry Veneer as MISSING first to calculate correlation

train_raw.MasVnrType.fillna('MISSING', inplace=True)
# Remove MiscFeature

print(train_raw.SalePrice.mean())

print(train_raw.groupby(['MiscFeature']).agg({'SalePrice':[np.mean, np.size]}))

sns.boxplot(x='MiscFeature', y='SalePrice', data = train_raw[['MiscFeature','SalePrice']])
# Only one data record with missing Electrical system

# find the most correlated field to Electrical, then impute based on the correlated fields

print(train_raw.groupby(['Electrical']).agg({'SalePrice':[np.mean, np.size]}))

train_raw[train_raw.Electrical.isna()]

# Fill Electrical as MISSING first to calculate correlation

train_raw.Electrical.fillna('MISSING', inplace=True)
train_raw.isna().mean(axis=0).sort_values(ascending=False)[:5]
def theil_u (x, y):

    e_x = drv.entropy(x)

    ce_xy = drv.entropy_conditional(x, y)

    if e_x == 0:

        return 1

    else:

        return (e_x - ce_xy)/e_x
def correlation_ratio (x, y):

    var = np.var(y)*len(y)

    group_var = np.var(pd.DataFrame({'x':x, 'y':y}).groupby(['x']).agg({'y':np.mean})) * len(np.unique(x))

    return np.sqrt(group_var/var)
categorical_feature = ['MSSubClass','MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood'

                       ,'Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType'

                      ,'ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating'

                      ,'HeatingQC','CentralAir','Electrical','KitchenQual','Functional','GarageType','GarageFinish'

                      ,'GarageCond','PavedDrive','SaleType']

correlation_ratio_mat = np.zeros(len(categorical_feature))

for i in range(len(categorical_feature)):

    correlation_ratio_mat[i] = correlation_ratio(train_raw[categorical_feature[i]], train_raw.SalePrice)
# Top 15 categorical features, that are most correlated to SalePrice

plt.figure(figsize=(12, 10))

plt.bar(np.arange(len(categorical_feature)), correlation_ratio_mat, align='center', alpha=0.5)

plt.xticks(np.arange(len(categorical_feature)), categorical_feature, rotation=45)

print([categorical_feature[i] for i in correlation_ratio_mat.argsort()[-15:][::-1]])
numeric_feature = ['LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2'

                   ,'BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath'

                   ,'HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF'

                  ,'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','SalePrice']



plt.figure(figsize=(20, 20))

sns.heatmap(train_raw[numeric_feature].corr(), annot=True)
cat_feature_class={}

cat_feature_encode={}
# ExterCond

order = train_raw.groupby(['ExterCond']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').index

size = train_raw.groupby(['ExterCond']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').values

plt.figure(figsize = (8,8))

plt.xticks(rotation=45)

sns.boxplot(x='ExterCond', y='SalePrice', data=train_raw[['ExterCond','SalePrice']], order=order)



cat_feature_class['ExterCond'] = train_raw.groupby(['ExterCond']).agg({'SalePrice':np.mean}).sort_values(by='SalePrice').index

cat_feature_encode['ExterCond'] = dict(zip(cat_feature_class['ExterCond'], range(1,len(cat_feature_class['ExterCond'])+1)))

train_raw.ExterCond.replace(to_replace=cat_feature_encode['ExterCond'], inplace=True)
# MSZoning

order = train_raw.groupby(['MSZoning']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').index

size = train_raw.groupby(['MSZoning']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').values

plt.figure(figsize = (8,8))

plt.xticks(rotation=45)

sns.boxplot(x='MSZoning', y='SalePrice', data=train_raw[['MSZoning','SalePrice']], order=order)



mszn_class = train_raw.groupby(['MSZoning']).agg({'SalePrice':np.mean}).sort_values(by='SalePrice').index

mszn_encode = dict(zip(mszn_class, range(1,len(mszn_class)+1)))

train_raw.MSZoning.replace(to_replace=mszn_encode, inplace=True)
# BsmtExposure

order = train_raw.groupby(['BsmtExposure']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').index

size = train_raw.groupby(['BsmtExposure']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').values

plt.figure(figsize = (8,8))

plt.xticks(rotation=45)

sns.boxplot(x='BsmtExposure', y='SalePrice', data=train_raw[['BsmtExposure','SalePrice']], order=order)



cat_feature_class['BsmtExposure'] = train_raw.groupby(['BsmtExposure']).agg({'SalePrice':np.mean}).sort_values(by='SalePrice').index

cat_feature_encode['BsmtExposure'] = dict(zip(cat_feature_class['BsmtExposure'], range(1,len(cat_feature_class['BsmtExposure'])+1)))

train_raw.BsmtExposure.replace(to_replace=cat_feature_encode['BsmtExposure'], inplace=True)
# BsmtCond

order = train_raw.groupby(['BsmtCond']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').index

size = train_raw.groupby(['BsmtCond']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').values

plt.figure(figsize=(8,8))

plt.xticks(rotation=45)

sns.boxplot(x='BsmtCond', y='SalePrice', data=train_raw[['BsmtCond','SalePrice']], order = order)



cat_feature_class['BsmtCond'] = train_raw.groupby(['BsmtCond']).agg({'SalePrice':np.mean}).sort_values(by='SalePrice').index

cat_feature_encode['BsmtCond'] = dict(zip(cat_feature_class['BsmtCond'], range(1, len(cat_feature_class['BsmtCond'])+1)))

train_raw.BsmtCond.replace(to_replace=cat_feature_encode['BsmtCond'], inplace=True)
# GarageType

order = train_raw.groupby(['GarageType']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').index

size = train_raw.groupby(['GarageType']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').values

plt.figure(figsize=(8,8))

plt.xticks(rotation=45)

sns.boxplot(x='GarageType', y='SalePrice', data=train_raw[['GarageType','SalePrice']], order = order)



cat_feature_class['GarageType'] = train_raw.groupby(['GarageType']).agg({'SalePrice':np.mean}).sort_values(by='SalePrice').index

cat_feature_encode['GarageType'] = dict(zip(cat_feature_class['GarageType'], range(1, len(cat_feature_class['GarageType'])+1)))

train_raw.GarageType.replace(to_replace=cat_feature_encode['GarageType'], inplace=True)
# MSSubClass

order = train_raw.groupby(['MSSubClass']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').index

size = train_raw.groupby(['MSSubClass']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').values

plt.figure(figsize=(8, 8))

plt.xticks(rotation=45)

sns.boxplot(x='MSSubClass', y='SalePrice', data = train_raw[['MSSubClass','SalePrice']], order = order)



cat_feature_class['MSSubClass'] = train_raw.groupby(['MSSubClass']).agg({'SalePrice':np.mean}).sort_values(by='SalePrice').index

cat_feature_encode['MSSubClass'] = dict(zip(cat_feature_class['MSSubClass'], range(1,len(cat_feature_class['MSSubClass'])+1)))

train_raw.MSSubClass.replace(to_replace = cat_feature_encode['MSSubClass'], inplace=True)
# SaleType (Test data with value populated) - Group most rare 6 groups into one category

order = train_raw.groupby(['SaleType']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').index

size = train_raw.groupby(['SaleType']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').values

plt.figure(figsize=(8, 8))

plt.xticks(rotation=45)

sns.boxplot(x='SaleType', y='SalePrice', data = train_raw[['SaleType','SalePrice']], order = order)



train_raw.SaleType.replace(to_replace = {'Con': 'Oth', 'CWD':'Oth', 'ConLI': 'Oth','ConLw':'Oth', 'ConLD':'Oth'}, inplace=True)

order = train_raw.groupby(['SaleType']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').index

size = train_raw.groupby(['SaleType']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').values

plt.figure(figsize=(8, 8))

plt.xticks(rotation=45)

sns.boxplot(x='SaleType', y='SalePrice', data = train_raw[['SaleType','SalePrice']], order = order)



cat_feature_class['SaleType'] = train_raw.groupby(['SaleType']).agg({'SalePrice':np.mean}).sort_values(by='SalePrice').index

cat_feature_encode['SaleType'] = dict(zip(cat_feature_class['SaleType'], range(1,len(cat_feature_class['SaleType'])+1)))

train_raw.SaleType.replace(to_replace = cat_feature_encode['SaleType'], inplace=True)
# KitchenQual 

order = train_raw.groupby(['KitchenQual']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').index

size = train_raw.groupby(['KitchenQual']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').values

plt.figure(figsize=(8, 8))

plt.xticks(rotation=45)

sns.boxplot(x='KitchenQual', y='SalePrice', data = train_raw[['KitchenQual','SalePrice']], order = order)



cat_feature_class['KitchenQual'] = train_raw.groupby(['KitchenQual']).agg({'SalePrice':np.mean}).sort_values(by='SalePrice').index

cat_feature_encode['KitchenQual'] = dict(zip(cat_feature_class['KitchenQual'], range(1,len(cat_feature_class['KitchenQual'])+1)))

train_raw.KitchenQual.replace(to_replace = cat_feature_encode['KitchenQual'], inplace=True)
# BsmtQual 

order = train_raw.groupby(['BsmtQual']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').index

size = train_raw.groupby(['BsmtQual']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').values

plt.figure(figsize=(8, 8))

plt.xticks(rotation=45)

sns.boxplot(x='BsmtQual', y='SalePrice', data = train_raw[['BsmtQual','SalePrice']], order = order)



cat_feature_class['BsmtQual'] = train_raw.groupby(['BsmtQual']).agg({'SalePrice':np.mean}).sort_values(by='SalePrice').index

cat_feature_encode['BsmtQual'] = dict(zip(cat_feature_class['BsmtQual'], range(1,len(cat_feature_class['BsmtQual'])+1)))

train_raw.BsmtQual.replace(to_replace = cat_feature_encode['BsmtQual'], inplace=True)
# ExterQual 

order = train_raw.groupby(['ExterQual']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').index

size = train_raw.groupby(['ExterQual']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').values

plt.figure(figsize=(8, 8))

plt.xticks(rotation=45)

sns.boxplot(x='ExterQual', y='SalePrice', data = train_raw[['ExterQual','SalePrice']], order = order)



cat_feature_class['ExterQual'] = train_raw.groupby(['ExterQual']).agg({'SalePrice':np.mean}).sort_values(by='SalePrice').index

cat_feature_encode['ExterQual'] = dict(zip(cat_feature_class['ExterQual'], range(1,len(cat_feature_class['ExterQual'])+1)))

train_raw.ExterQual.replace(to_replace = cat_feature_encode['ExterQual'], inplace=True)
# RoofMatl - Most values falls with CompShg, consider to remove the feature

order = train_raw.groupby(['RoofMatl']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').index

size = train_raw.groupby(['RoofMatl']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').values

plt.figure(figsize=(8, 8))

plt.xticks(rotation=45)

sns.boxplot(x='RoofMatl', y='SalePrice', data = train_raw[['RoofMatl','SalePrice']], order = order)
# Exterior2nd - Encode with target mean ordinal order, group 5 most minority as one category.

# Maybe remove, to check correlation with Exterior1st

order = train_raw.groupby(['Exterior2nd']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').index

size = train_raw.groupby(['Exterior2nd']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').values

plt.figure(figsize=(8, 8))

plt.xticks(rotation=45)

sns.boxplot(x='Exterior2nd', y='SalePrice', data = train_raw[['Exterior2nd','SalePrice']], order = order)



train_raw.Exterior2nd.replace(to_replace = {'CBlock': 'Other', 'Stone':'Other', 'AsphShn': 'Other','Brk Cmn':'Other'}, inplace=True)

order = train_raw.groupby(['Exterior2nd']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').index

size = train_raw.groupby(['Exterior2nd']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').values

plt.figure(figsize=(8, 8))

plt.xticks(rotation=45)

sns.boxplot(x='Exterior2nd', y='SalePrice', data = train_raw[['Exterior2nd','SalePrice']], order = order)



ext2nd_class = train_raw.groupby(['Exterior2nd']).agg({'SalePrice':np.mean}).sort_values(by='SalePrice').index

ext2nd_encode = dict(zip(ext2nd_class, range(1,len(ext2nd_class)+1)))

train_raw.Exterior2nd.replace(to_replace = ext2nd_encode, inplace=True)
# Exterior1st - Encode with target mean ordinal order, group 4 most minority as one category

order = train_raw.groupby(['Exterior1st']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').index

size = train_raw.groupby(['Exterior1st']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').values

plt.figure(figsize=(8, 8))

plt.xticks(rotation=45)

sns.boxplot(x='Exterior1st', y='SalePrice', data = train_raw[['Exterior1st','SalePrice']], order = order)



train_raw.Exterior1st.replace(to_replace = {'AsphShn': 'Other', 'CBlock':'Other', 'ImStucc': 'Other','BrkComm':'Other'}, inplace=True)

order = train_raw.groupby(['Exterior1st']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').index

size = train_raw.groupby(['Exterior1st']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').values

plt.figure(figsize=(8, 8))

plt.xticks(rotation=45)

sns.boxplot(x='Exterior1st', y='SalePrice', data = train_raw[['Exterior1st','SalePrice']], order = order)



cat_feature_class['Exterior1st'] = train_raw.groupby(['Exterior1st']).agg({'SalePrice':np.mean}).sort_values(by='SalePrice').index

cat_feature_encode['Exterior1st'] = dict(zip(cat_feature_class['Exterior1st'], range(1,len(cat_feature_class['Exterior1st'])+1)))

train_raw.Exterior1st.replace(to_replace = cat_feature_encode['Exterior1st'], inplace=True)
# Condition2 -- Most values falls with Norm, consider to remove the feature

order = train_raw.groupby(['Condition2']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').index

size = train_raw.groupby(['Condition2']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').values

plt.figure(figsize=(8, 8))

plt.xticks(rotation=45)

sns.boxplot(x='Condition2', y='SalePrice', data = train_raw[['Condition2','SalePrice']], order = order)
# Neighborhood - Encode with target mean ordinal order

order = train_raw.groupby(['Neighborhood']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').index

size = train_raw.groupby(['Neighborhood']).agg({'SalePrice':np.size}).sort_values(by='SalePrice').values

plt.figure(figsize=(8, 8))

plt.xticks(rotation=45)

sns.boxplot(x='Neighborhood', y='SalePrice', data = train_raw[['Neighborhood','SalePrice']], order = order)

cat_feature_class['Neighborhood'] = train_raw.groupby(['Neighborhood']).agg({'SalePrice':np.mean}).sort_values(by='SalePrice').index

cat_feature_encode['Neighborhood'] = dict(zip(cat_feature_class['Neighborhood'], range(1,len(cat_feature_class['Neighborhood'])+1)))

train_raw.Neighborhood.replace(to_replace = cat_feature_encode['Neighborhood'], inplace=True)
#cat_feature_15 = [categorical_feature[i] for i in correlation_ratio_mat.argsort()[-15:][::-1]]

# Removed Condition2, RoofMat features

cat_feature_15 = ['Neighborhood', 'Exterior1st', 'Exterior2nd', 'ExterQual', 'BsmtQual', 'KitchenQual', 'SaleType'

                  , 'MSSubClass', 'GarageType', 'BsmtCond', 'BsmtExposure', 'MSZoning', 'ExterCond']

# num_feature_15 = train_raw[numeric_feature].corr()['SalePrice'].sort_values(ascending=False)[1:16].index

num_feature_15 =['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF','1stFlrSF', 'FullBath'

                 , 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd','GarageYrBlt', 'MasVnrArea', 'Fireplaces', 'BsmtFinSF1', 'LotFrontage']
theil_u_mat = np.zeros((len(cat_feature_15),len(cat_feature_15)))

for i in range(len(cat_feature_15)):

    for j in range(len(cat_feature_15)):

        theil_u_mat[i][j] = theil_u(train_raw[cat_feature_15[i]], train_raw[cat_feature_15[j]])          

plt.figure(figsize=(20, 20))

sns.heatmap(pd.DataFrame(data=theil_u_mat, columns = cat_feature_15, index = cat_feature_15), annot=True)   

# Remove highly correlated fields Exterior2nd, MSZoning
plt.figure(figsize=(20, 20))

sns.heatmap(train_raw[num_feature_15].corr(), annot=True)

# Remove highly correlated fields GarageArea, 1stFlrSF, TotRmsAbvGrd, GarageYrBlt
corr_cat_num_mat = np.zeros((len(cat_feature_15),len(num_feature_15)))

for i in range(len(cat_feature_15)):

    for j in range(len(num_feature_15)):

        corr_cat_num_mat[i][j] = correlation_ratio(train_raw[cat_feature_15[i]], train_raw[num_feature_15[j]])          

plt.figure(figsize=(20, 20))

sns.heatmap(pd.DataFrame(data=corr_cat_num_mat, columns = num_feature_15, index = cat_feature_15), annot=True)   
# Age = YrSold - YearRemodAdd, hence removing YearRemodAdd, and add Age.

# As per validation result, use Age is better than use YearRemodAdd

train_raw['Age'] = train_raw.YrSold-train_raw.YearRemodAdd

# Assess GFC impact on house price -- Not much impact though (why?)

train_raw['PseudoDaySold'] = 1

train_raw['DateSold'] = pd.to_datetime({'year':train_raw.YrSold, 'month':train_raw.MoSold, 'day':train_raw.PseudoDaySold})

order = sorted(train_raw.DateSold.unique())

plt.figure(figsize =(15,15))

plt.xticks(rotation=45)

sns.boxplot(x='DateSold',y='SalePrice',data=train_raw[['DateSold','SalePrice']], order=order)
final_feature = ['Neighborhood', 'Exterior1st', 'ExterQual', 'BsmtQual', 'KitchenQual', 'SaleType'

                  , 'MSSubClass', 'GarageType', 'BsmtCond', 'BsmtExposure', 'ExterCond'

                , 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath'

                 , 'YearBuilt', 'Age', 'MasVnrArea', 'Fireplaces', 'BsmtFinSF1', 'LotFrontage']



# Train validation split

X = np.array(train_raw[final_feature])

y = np.log10(np.array(train_raw.SalePrice)+1) # Convert to log10

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)



# Model tuning for the best min_child_weight, max_depth

tuning_dict = {}

max_depth_list = np.ceil(np.random.rand(9)*10).astype(int)

min_child_weight_list = np.power(10,np.random.rand(9)*2.5).astype(int)

for i in range(9):

    model = xgboost.XGBRegressor(learning_rate=0.1, n_estimators=1000, min_child_weight =min_child_weight_list[i], max_depth= max_depth_list[i], seed=42) #min_child_weight: control regularization

    model.fit(X_train, y_train, eval_set=[(X_train, y_train),(X_val, y_val)], eval_metric='rmse' )

    val_rmsle = model.evals_result()['validation_1']['rmse'][999]

    tuning_dict[i]={'max_depth':max_depth_list[i], 'min_child_weight':min_child_weight_list[i], 'val_rmsle':val_rmsle}
tuning_dict # Choose max_depth as 2, min_child_weight as 5
test_raw = pd.read_csv('../input/test.csv')

test_raw.info()
# Compare test distribution vs train distribution 

# Almost all distribution looks good

test_raw.describe()
train_raw1 = pd.read_csv('../input/train.csv')

train_raw1.describe()
test_raw.describe(include='O')
train_raw1.describe(include='O')
test_raw[cat_features].isna().mean().sort_values(ascending =False)
# Impute Garage

test_raw[(test_raw.GarageType.isna()&(test_raw.GarageYrBlt.notna()|test_raw.GarageFinish.notna()|(test_raw.GarageCars != 0) |(test_raw.GarageArea != 0)|test_raw.GarageQual.notna()

                                      |test_raw.GarageCond.notna()))][['GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond']]

test_raw.GarageType.fillna('NoGarage', inplace=True)

# Impute Basement

test_raw[(test_raw.BsmtQual.isna()&(test_raw.BsmtCond.notna()|test_raw.BsmtExposure.notna()|(test_raw.BsmtFinSF2 != 0) |(test_raw.BsmtFinSF1 != 0)|test_raw.BsmtFinType2.notna()

                                      |test_raw.BsmtFinType2.notna()|(test_raw.BsmtUnfSF != 0)|(test_raw.TotalBsmtSF != 0)|(test_raw.BsmtFullBath != 0)|(test_raw.BsmtHalfBath != 0)))][

    ['BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','GarageCond','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']]

test_raw.at[660,'BsmtQual']= 'NoBasement'

test_raw.at[728,'BsmtQual']= 'NoBasement'

test_raw.at[757,'BsmtQual']= 'Fa'

test_raw.at[758,'BsmtQual']= 'TA'

test_raw[(test_raw.BsmtExposure.isna()&(test_raw.BsmtQual.notna()|test_raw.BsmtCond.notna()|(test_raw.BsmtFinSF2 != 0) |(test_raw.BsmtFinSF1 != 0)|test_raw.BsmtFinType2.notna()

                                      |test_raw.BsmtFinType2.notna()|(test_raw.BsmtUnfSF != 0)|(test_raw.TotalBsmtSF != 0)|(test_raw.BsmtFullBath != 0)|(test_raw.BsmtHalfBath != 0)))][

    ['BsmtQual','BsmtCond','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','GarageCond','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']]

test_raw.at[27,'BsmtExposure']= test_raw[(test_raw.BsmtQual=='Gd') & (test_raw.BsmtCond=='TA')].BsmtExposure.mode()[0]

test_raw.at[660,'BsmtExposure']= 'NoBasement'

test_raw.at[728,'BsmtExposure']= 'NoBasement'

test_raw.at[888,'BsmtExposure']= test_raw[(test_raw.BsmtQual=='Gd') & (test_raw.BsmtCond=='TA')].BsmtExposure.mode()[0]

test_raw[(test_raw.BsmtCond.isna()&(test_raw.BsmtQual.notna()|test_raw.BsmtExposure.notna()|(test_raw.BsmtFinSF2 != 0) |(test_raw.BsmtFinSF1 != 0)|test_raw.BsmtFinType2.notna()

                                      |test_raw.BsmtFinType2.notna()|(test_raw.BsmtUnfSF != 0)|(test_raw.TotalBsmtSF != 0)|(test_raw.BsmtFullBath != 0)|(test_raw.BsmtHalfBath != 0)))][

    ['BsmtQual','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','GarageCond','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']]

test_raw.at[580,'BsmtCond']= 'Gd'

test_raw.at[660,'BsmtCond']= 'NoBasement'

test_raw.at[725,'BsmtCond']= 'TA'

test_raw.at[728,'BsmtCond']= 'NoBasement'

test_raw.at[1064,'BsmtCond']= 'TA'



test_raw.BsmtQual.fillna('NoBasement', inplace=True)

test_raw.BsmtExposure.fillna('NoBasement', inplace=True)

test_raw.BsmtCond.fillna('NoBasement', inplace=True)



# Impute KitchenQual by most frequent value

test_raw.at[95,'KitchenQual'] = test_raw.KitchenQual.mode()[0]

# Impute Exterior1st, based on most frequent value of the same ExterQual and ExterCond

test_raw.at[691,'Exterior1st']=test_raw[(test_raw.ExterQual=='TA')  & (test_raw.ExterCond=='TA')].Exterior1st.mode()[0]

# Impute SaleType, based on most frequent value of the same YrSold

test_raw.at[1029,'SaleType'] = test_raw[test_raw.YrSold==2007].SaleType.mode()[0]

# Test data, group minorities 

test_raw.Exterior1st.replace(to_replace = {'AsphShn': 'Other', 'CBlock':'Other', 'ImStucc': 'Other','BrkComm':'Other'}, inplace=True)

test_raw.SaleType.replace(to_replace = {'Con': 'Oth', 'CWD':'Oth', 'ConLI': 'Oth','ConLw':'Oth', 'ConLD':'Oth'}, inplace=True)

# Encode Test categorical data

cat_features = ['Neighborhood', 'Exterior1st', 'ExterQual', 'BsmtQual', 'KitchenQual', 'SaleType', 'MSSubClass', 'GarageType', 'BsmtCond', 'BsmtExposure', 'ExterCond']

for col in cat_features:

    for cat in test_raw[col].unique():

        if cat not in cat_feature_class[col]:

            print('column: '+str(col)+' value: '+str(cat)+' not in train data')

for col in cat_features:

    test_raw[col].replace(to_replace=cat_feature_encode[col], inplace=True)

# For MSSubClass = 150, never appears in train data

test_raw.MSSubClass.replace(to_replace={150:0}, inplace=True)
# Feature engineer for Test data

test_raw['Age'] = test_raw.YrSold-test_raw.YearRemodAdd
# Prediction for Test data

final_feature = ['Neighborhood', 'Exterior1st', 'ExterQual', 'BsmtQual', 'KitchenQual', 'SaleType'

                  , 'MSSubClass', 'GarageType', 'BsmtCond', 'BsmtExposure', 'ExterCond'

                , 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath'

                 , 'YearBuilt', 'Age', 'MasVnrArea', 'Fireplaces', 'BsmtFinSF1', 'LotFrontage']



# Train validation split

model = xgboost.XGBRegressor(learning_rate=0.1, n_estimators=1000, min_child_weight =5, max_depth=2, seed=42) #min_child_weight: control regularization

model.fit(X, y, eval_metric='rmse' )



y_test = model.predict(np.array(test_raw[final_feature]))

submission = pd.DataFrame(data={'Id':test_raw.Id,'SalePrice':np.power(10,y_test)-1})

submission.to_csv('LXH_submission.csv',index=False)
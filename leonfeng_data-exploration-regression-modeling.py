import pandas as pd

import numpy as np

import matplotlib.pylab as plt

plt.style.use('bmh')

%matplotlib inline

import seaborn as sns



from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import scale



from sklearn.linear_model import LinearRegression, Ridge, RidgeCV

from sklearn.ensemble import GradientBoostingRegressor
pd.set_option('display.max_columns', 85)

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print('train shape:', train.shape, '\n', 'test shape:', test.shape)

train.head()
missing_numeric = pd.concat([train.isnull().sum(), test.isnull().sum()], axis=1, keys=['train', 'test'])

missing_numeric = missing_numeric[(missing_numeric['train']>0) | (missing_numeric['test']>0)]

missing_numeric.sort_values(by=['train', 'test'], ascending=False)
# Drop the features which I'm not interested in 

feature_drop = ['PoolQC', 'MiscFeature', 'Fence', 'FireplaceQu', 'LotFrontage', 'GarageYrBlt', 'MoSold', 'YrSold', 

                'LowQualFinSF', 'MiscVal', 'PoolArea']

datasets = [train, test]

for df in datasets:

    df.drop(feature_drop, axis=1, inplace=True)

    df.loc[df['Alley'].isnull(), 'Alley'] = 'NoAlley'

# If a house has no garage, it will have missing value on the 'Garage related' features, so just fill NaNs with 'NoGarage'.

    df.loc[df['GarageCond'].isnull(), 'GarageCond'] = 'NoGarage'

    df.loc[df['GarageQual'].isnull(), 'GarageQual'] = 'NoGarage'

    df.loc[df['GarageType'].isnull(), 'GarageType'] = 'NoGarage'

    df.loc[df['GarageFinish'].isnull(), 'GarageFinish'] = 'NoGarage'

    

# If a house has no basement, it will have missing value on the 'basement related' features, so just fill NaNs with 'NoBsmt'.    

    df.loc[df['BsmtExposure'].isnull(), 'BsmtExposure'] = 'NoBsmt'

    df.loc[df['BsmtFinType2'].isnull(), 'BsmtFinType2'] = 'NoBsmt'

    df.loc[df['BsmtCond'].isnull(), 'BsmtCond'] = 'NoBsmt'

    df.loc[df['BsmtQual'].isnull(), 'BsmtQual'] = 'NoBsmt'

    df.loc[df['BsmtFinType1'].isnull(), 'BsmtFinType1'] = 'NoBsmt'

    

# Masonry veneer feature: just fill with 'None' if there is no Masonry veneer.    

    df.loc[df['MasVnrType'].isnull(), 'MasVnrType'] = 'None'

    df.loc[df['MasVnrArea'].isnull(), 'MasVnrArea'] = 0

       

train['Electrical'].fillna(train['Electrical'].mode()[0], inplace=True)



test_numeric_missing = ['BsmtFullBath', 'BsmtHalfBath', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'GarageArea', 'GarageCars', 'TotalBsmtSF']

test_categorical_missing = ['MSZoning', 'Functional', 'Utilities', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'SaleType']



for i in test_numeric_missing:

    test[i].fillna(0, inplace=True)

for j in test_categorical_missing:

    test[j].fillna(test[j].mode()[0], inplace=True)



# Check the missing values again for datasets

missing_numeric = pd.concat([train.isnull().sum(), test.isnull().sum()], axis=1, keys=['train', 'test'])

missing_numeric = missing_numeric[(missing_numeric['train']>0) | (missing_numeric['test']>0)]

missing_numeric.sort_values(by=['train', 'test'], ascending=False)
train.select_dtypes(exclude=[object]).describe()
print(train['SalePrice'].describe(), '\n')

print('Before Transformation Skew: ', train['SalePrice'].skew())



target = np.log1p(train['SalePrice'])

print('Log Transformation Skew: ', target.skew())



plt.rcParams['figure.figsize'] = (12, 5)

target_log_tran = pd.DataFrame({'befrore transformation':train['SalePrice'], 'log transformation': target})

target_log_tran.hist()
skewness = pd.DataFrame({'Skewness':train.select_dtypes(exclude=[object]).skew()})



print(skewness[skewness['Skewness']>0.8].sort_values(by='Skewness'), '\n')  

print(skewness[skewness['Skewness']>0.8].sort_values(by='Skewness').index.tolist())
skews = ['2ndFlrSF', 'BsmtUnfSF', 'GrLivArea', '1stFlrSF', 'MSSubClass', 'TotalBsmtSF', 'WoodDeckSF', 'BsmtFinSF1', 'OpenPorchSF', 

         'MasVnrArea', 'EnclosedPorch', 'BsmtHalfBath', 'ScreenPorch', 'BsmtFinSF2', 'KitchenAbvGr', '3SsnPorch', 'LotArea']

for df in datasets:

    for s in skews:

        df[s] = np.log1p(df[s])
corr = train.select_dtypes(exclude=[object]).corr()

print(corr['SalePrice'].sort_values(ascending=False)[:22], '\n')

print(corr['SalePrice'].sort_values(ascending=False)[-5:])
numeric_data = train[['SalePrice', 'OverallQual', 'GrLivArea', 'GarageArea', 'GarageCars', 'TotalBsmtSF', '1stFlrSF', 

                             'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'Fireplaces', 'BsmtFinSF1',

                            'WoodDeckSF', '2ndFlrSF', 'OpenPorchSF', 'HalfBath', 'LotArea', 'BsmtFullBath', 'BsmtUnfSF']]



corr = numeric_data.corr()

plt.figure(figsize=(12, 12))

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, vmax=1, square=True, annot=True, mask=mask, cbar=False, linewidths=0.1)

plt.xticks(rotation=45)
numeric_data_select = train[['SalePrice', 'OverallQual', 'GrLivArea', 'GarageArea', 'FullBath', 'TotalBsmtSF', 

                                    'YearBuilt', 'MasVnrArea', 'Fireplaces', 'BsmtFinSF1', 'WoodDeckSF', 'OpenPorchSF',

                                    'HalfBath', 'LotArea']]

corr_select = numeric_data_select.corr()

plt.figure(figsize=(8, 8))

mask = np.zeros_like(corr_select)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr_select, vmax=1, square=True, annot=True, mask=mask, cbar=False, linewidths=0.1)

plt.xticks(rotation=45)
sns.pairplot(numeric_data_select, size=2)
plt.rcParams['figure.figsize'] = (12, 4)

plt.subplot(121)

sns.boxplot(train['OverallQual'], target)

plt.subplot(122)

sns.boxplot(train['FullBath'], target)
plt.subplot(121)

plt.scatter(train['GrLivArea'], target)

plt.xlabel('GrLivArea')

plt.ylabel('SalePrice')

plt.subplot(122)

plt.scatter(train['GarageArea'], target)

plt.xlabel('GarageArea')

plt.ylabel('SalePrice')
plt.subplot(121)

plt.scatter(train['TotalBsmtSF'], target)

plt.xlabel('TotalBsmtSF')

plt.ylabel('SalePrice')



plt.subplot(122)

plt.scatter(train['MasVnrArea'], target)

plt.xlabel('MasVnrArea')

plt.ylabel('SalePrice')
plt.subplot(121)

sns.boxplot(train['Fireplaces'], target)



plt.subplot(122)

plt.scatter(train['YearBuilt'], target)

plt.xlabel('YearBuilt')
plt.subplot(121)

plt.scatter(train['BsmtFinSF1'], target)

plt.xlabel('BsmtFinSF1')

plt.ylabel('SalePrice')



plt.subplot(122)

plt.scatter(train['WoodDeckSF'], target)

plt.xlabel('WoodDeckSF')

plt.ylabel('SalePrice')
index_remove = train[train['GrLivArea'] > 8.5].index.tolist()+train[train['GarageArea'] > 1200].index.tolist()+train[train['TotalBsmtSF'] > 8.2].index.tolist()+train[train['BsmtFinSF1'] > 8].index.tolist()

index_remove = list(set(index_remove))  # remove duplicate values

index_remove.append(523)

print(index_remove)          



train = train.drop(train.index[index_remove], axis=0)

train = train[train['SalePrice'] <=550000]
categorical_data = train.select_dtypes(include=[object])

categorical_data.describe()
plt.rcParams['figure.figsize'] = (12, 7)

plt.subplot(221)

sns.boxplot(train['ExterQual'], target)

plt.subplot(222)

sns.boxplot(train['BsmtQual'], target)

plt.subplot(223)

sns.boxplot(train['BsmtExposure'], target)

plt.subplot(224)

sns.boxplot(train['GarageFinish'], target)
plt.subplot(221)

sns.boxplot(train['SaleCondition'], target)

plt.subplot(222)

sns.boxplot(train['CentralAir'], target)

plt.subplot(223)

sns.boxplot(train['KitchenQual'], target)
train_ExterQual_dummy = pd.get_dummies(train['ExterQual'], prefix='ExterQual')

test_ExterQual_dummy = pd.get_dummies(test['ExterQual'], prefix='ExterQual')



train_BsmtQual_dummy = pd.get_dummies(train['BsmtQual'], prefix='BsmtQual')

test_BsmtQual_dummy = pd.get_dummies(test['BsmtQual'], prefix='BsmtQual')



train_BsmtExposure_dummy = pd.get_dummies(train['BsmtExposure'], prefix='BsmtExposure')

test_BsmtExposure_dummy = pd.get_dummies(test['BsmtExposure'], prefix='BsmtExposure')



train_GarageFinish_dummy = pd.get_dummies(train['GarageFinish'], prefix='GarageFinish')

test_GarageFinish_dummy = pd.get_dummies(test['GarageFinish'], prefix='GarageFinish')



train_SaleCondition_dummy = pd.get_dummies(train['SaleCondition'], prefix='SaleCondition')

test_SaleCondition_dummy = pd.get_dummies(test['SaleCondition'], prefix='SaleCondition')



train_CentralAir_dummy = pd.get_dummies(train['CentralAir'], prefix='CentralAir')

test_CentralAir_dummy = pd.get_dummies(test['CentralAir'], prefix='CentralAir')



train_KitchenQual_dummy = pd.get_dummies(train['KitchenQual'], prefix='KitchenQual')

test_KitchenQual_dummy = pd.get_dummies(test['KitchenQual'], prefix='KitchenQual')
# Define a model evaluation function by outputing R2 score and mean squared error. (using 10-fold cross validation)

def model_eval(model):

    model_fit = model.fit(X, y)

    R2 = cross_val_score(model_fit, X, y, cv=10 , scoring='r2').mean()

    MSE = -cross_val_score(lr, X, y, cv=10 , scoring='neg_mean_squared_error').mean()

    print('R2 Score:', R2, '|', 'MSE:', MSE)
data = train.select_dtypes(exclude=[object])

y = np.log1p(data['SalePrice'])

X = data.drop(['Id', 'SalePrice'], axis=1)

X = pd.concat([X, train_ExterQual_dummy, train_BsmtQual_dummy, train_GarageFinish_dummy, train_BsmtExposure_dummy,

              train_SaleCondition_dummy, train_CentralAir_dummy, train_KitchenQual_dummy], axis=1)
lr = LinearRegression()

ri = Ridge(alpha=0.1, normalize=False)

ricv = RidgeCV(cv=5)

gdb = GradientBoostingRegressor(n_estimators=200)
for model in [lr, ri, ricv, gdb]:

    model_eval(model)
test_id = test['Id']

test = test.select_dtypes(exclude=[object]).drop('Id', axis=1)

test = pd.concat([test, test_ExterQual_dummy, test_BsmtQual_dummy, test_GarageFinish_dummy, test_BsmtExposure_dummy,

              test_SaleCondition_dummy, test_CentralAir_dummy, test_KitchenQual_dummy], axis=1)
pred = ri.predict(test)
pred = np.expm1(pred)

prediction = pd.DataFrame({'Id':test_id, 'SalePrice':pred})

prediction.to_csv('Prediction1.csv', index=False)

prediction.head()
plt.scatter(cross_val_predict(lr, X, y), y)

plt.xlabel('Predicted Values')

plt.ylabel('True Values')
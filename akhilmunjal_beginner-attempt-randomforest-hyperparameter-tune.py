import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestRegressor
raw_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

pd.set_option("display.max_columns",0)

raw_data
raw_data.shape
numeric = raw_data.select_dtypes({'int64','float64'}).columns

numeric
categorical = raw_data.select_dtypes({'object'}).columns

categorical
#Number of numeric columns

raw_data.select_dtypes({'int64','float64'}).shape[1]
#Number of categorical columns

raw_data.select_dtypes({'object'}).shape[1]
raw_data['SalePrice'].describe()
sns.distplot(raw_data['SalePrice'])
raw_data['SalePrice'].skew()
raw_data['SalePrice'].kurt()
#correlation matrix

corrmat = raw_data.corr()

plt.subplots(figsize=(10, 10))

sns.heatmap(corrmat, square=True);
k = 10

most_correlated = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

print(most_correlated)
factors = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',

            'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
#Checking for duplicate values:

dup_data = raw_data[raw_data.duplicated()]

dup_data.shape[0]
total = pd.isnull(raw_data).sum().sort_values(ascending=False)

percentage = ((pd.isnull(raw_data).sum() / pd.isnull(raw_data).count()).sort_values(ascending=False))*100

no_values = pd.concat([total,percentage],axis = 1, keys = ['Total', 'Percent'])
no_values.head(19)
raw_data_without_null = raw_data.drop((no_values[no_values['Total'] > 1]).index,1)
raw_data_without_null = raw_data_without_null.dropna()
pd.isnull(raw_data_without_null).sum().max()
raw_data_without_null.shape
sns.boxplot(raw_data_without_null['OverallQual'])

q1 = raw_data_without_null['OverallQual'].quantile(.25)

q3 = raw_data_without_null['OverallQual'].quantile(.75)

iqr = q3 - q1

floor = q1 - 3*iqr

cap = q3 + 3*iqr

print('Floor = {}, Capping = {}'.format(floor,cap))

raw_data_without_null = raw_data_without_null[(raw_data_without_null['OverallQual'] > floor) & (raw_data_without_null['OverallQual'] < cap)]
sns.boxplot(raw_data_without_null['GrLivArea'])

q1 = raw_data_without_null['GrLivArea'].quantile(.25)

q3 = raw_data_without_null['GrLivArea'].quantile(.75)

iqr = q3 - q1

floor = q1 - 3*iqr

cap = q3 + 3*iqr

print('Floor = {}, Capping = {}'.format(floor,cap))

raw_data_without_null = raw_data_without_null[(raw_data_without_null['GrLivArea'] > floor) & (raw_data_without_null['GrLivArea'] < cap)]
sns.boxplot(raw_data_without_null['GarageCars'])

q1 = raw_data_without_null['GarageCars'].quantile(.25)

q3 = raw_data_without_null['GarageCars'].quantile(.75)

iqr = q3 - q1

floor = q1 - 3*iqr

cap = q3 + 3*iqr

print('Floor = {}, Capping = {}'.format(floor,cap))

raw_data_without_null = raw_data_without_null[(raw_data_without_null['GarageCars'] > floor) & (raw_data_without_null['GarageCars'] < cap)]
sns.boxplot(raw_data_without_null['GarageArea'])

q1 = raw_data_without_null['GarageArea'].quantile(.25)

q3 = raw_data_without_null['GarageArea'].quantile(.75)

iqr = q3 - q1

floor = q1 - 3*iqr

cap = q3 + 3*iqr

print('Floor = {}, Capping = {}'.format(floor,cap))

raw_data_without_null = raw_data_without_null[(raw_data_without_null['GarageArea'] > floor) & (raw_data_without_null['GarageArea'] < cap)]
sns.boxplot(raw_data_without_null['TotalBsmtSF'])

q1 = raw_data_without_null['TotalBsmtSF'].quantile(.25)

q3 = raw_data_without_null['TotalBsmtSF'].quantile(.75)

iqr = q3 - q1

floor = q1 - 3*iqr

cap = q3 + 3*iqr

print('Floor = {}, Capping = {}'.format(floor,cap))

raw_data_without_null = raw_data_without_null[(raw_data_without_null['TotalBsmtSF'] > floor) & (raw_data_without_null['TotalBsmtSF'] < cap)]
sns.boxplot(raw_data_without_null['1stFlrSF'])

q1 = raw_data_without_null['1stFlrSF'].quantile(.25)

q3 = raw_data_without_null['1stFlrSF'].quantile(.75)

iqr = q3 - q1

floor = q1 - 3*iqr

cap = q3 + 3*iqr

print('Floor = {}, Capping = {}'.format(floor,cap))

raw_data_without_null = raw_data_without_null[(raw_data_without_null['1stFlrSF'] > floor) & (raw_data_without_null['1stFlrSF'] < cap)]
sns.boxplot(raw_data_without_null['FullBath'])

q1 = raw_data_without_null['FullBath'].quantile(.25)

q3 = raw_data_without_null['FullBath'].quantile(.75)

iqr = q3 - q1

floor = q1 - 3*iqr

cap = q3 + 3*iqr

print('Floor = {}, Capping = {}'.format(floor,cap))

raw_data_without_null = raw_data_without_null[(raw_data_without_null['FullBath'] > floor) & (raw_data_without_null['FullBath'] < cap)]
sns.boxplot(raw_data_without_null['TotRmsAbvGrd'])

q1 = raw_data_without_null['TotRmsAbvGrd'].quantile(.25)

q3 = raw_data_without_null['TotRmsAbvGrd'].quantile(.75)

iqr = q3 - q1

floor = q1 - 3*iqr

cap = q3 + 3*iqr

print('Floor = {}, Capping = {}'.format(floor,cap))

raw_data_without_null = raw_data_without_null[(raw_data_without_null['TotRmsAbvGrd'] > floor) & (raw_data_without_null['TotRmsAbvGrd'] < cap)]
sns.boxplot(raw_data_without_null['YearBuilt'])

q1 = raw_data_without_null['YearBuilt'].quantile(.25)

q3 = raw_data_without_null['YearBuilt'].quantile(.75)

iqr = q3 - q1

floor = q1 - 3*iqr

cap = q3 + 3*iqr

print('Floor = {}, Capping = {}'.format(floor,cap))

raw_data_without_null = raw_data_without_null[(raw_data_without_null['YearBuilt'] > floor) & (raw_data_without_null['YearBuilt'] < cap)]
raw_data_without_null.shape
data_cleaned = raw_data_without_null.copy()
data_use = data_cleaned[factors].copy()
f, (ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9) = plt.subplots(1,9, sharey = True, figsize = (30,5))

ax1.scatter(data_use['OverallQual'],data_use['SalePrice'])

ax1.set_title('OverallQual and SalePrice')



ax2.scatter(data_use['GrLivArea'],data_use['SalePrice'])

ax2.set_title('GrLivArea and SalePrice')



ax3.scatter(data_use['GarageCars'],data_use['SalePrice'])

ax3.set_title('GarageCars and SalePrice')



ax4.scatter(data_use['GarageArea'],data_use['SalePrice'])

ax4.set_title('GarageArea and SalePrice')



ax5.scatter(data_use['TotalBsmtSF'],data_use['SalePrice'])

ax5.set_title('TotalBsmtSF and SalePrice')



ax6.scatter(data_use['1stFlrSF'],data_use['SalePrice'])

ax6.set_title('1stFlrSF and SalePrice')



ax7.scatter(data_use['FullBath'],data_use['SalePrice'])

ax7.set_title('FullBath and SalePrice')



ax8.scatter(data_use['TotRmsAbvGrd'],data_use['SalePrice'])

ax8.set_title('TotRmsAbvGrd and SalePrice')



ax9.scatter(data_use['YearBuilt'],data_use['SalePrice'])

ax9.set_title('YearBuilt and SalePrice')
data_use['logPrice'] = np.log(data_use['SalePrice'])
before = raw_data['SalePrice'].skew()

after = data_use['logPrice'].skew()

print('Skewness before : {}, Skewness after : {}'.format(before,after))
sns.distplot(data_use['logPrice'])
f, (ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9) = plt.subplots(1,9, sharey = True, figsize = (30,5))

ax1.scatter(data_use['OverallQual'],data_use['logPrice'])

ax1.set_title('OverallQual and logPrice')



ax2.scatter(data_use['GrLivArea'],data_use['logPrice'])

ax2.set_title('GrLivArea and logPrice')



ax3.scatter(data_use['GarageCars'],data_use['logPrice'])

ax3.set_title('GarageCars and logPrice')



ax4.scatter(data_use['GarageArea'],data_use['logPrice'])

ax4.set_title('GarageArea and logPrice')



ax5.scatter(data_use['TotalBsmtSF'],data_use['logPrice'])

ax5.set_title('TotalBsmtSF and logPrice')



ax6.scatter(data_use['1stFlrSF'],data_use['logPrice'])

ax6.set_title('1stFlrSF and logPrice')



ax7.scatter(data_use['FullBath'],data_use['logPrice'])

ax7.set_title('FullBath and logPrice')



ax8.scatter(data_use['TotRmsAbvGrd'],data_use['logPrice'])

ax8.set_title('TotRmsAbvGrd and logPrice')



ax9.scatter(data_use['YearBuilt'],data_use['logPrice'])

ax9.set_title('YearBuilt and logPrice')
location = data_cleaned[['Neighborhood','Condition1']].copy()

data = data_cleaned[factors]

data_with_cat = data.join(location)

data_with_cat
data_with_cat['logPrice'] = np.log(data_with_cat['SalePrice'])

data_with_cat = data_with_cat.drop(['SalePrice'],axis=1)
data_with_dummies = pd.get_dummies(data_with_cat, drop_first = True)
cols = ['logPrice','OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',

       'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt',

        'Neighborhood_Blueste', 'Neighborhood_BrDale',

       'Neighborhood_BrkSide', 'Neighborhood_ClearCr',

       'Neighborhood_CollgCr', 'Neighborhood_Crawfor',

       'Neighborhood_Edwards', 'Neighborhood_Gilbert',

       'Neighborhood_IDOTRR', 'Neighborhood_MeadowV',

       'Neighborhood_Mitchel', 'Neighborhood_NAmes',

       'Neighborhood_NPkVill', 'Neighborhood_NWAmes',

       'Neighborhood_NoRidge', 'Neighborhood_NridgHt',

       'Neighborhood_OldTown', 'Neighborhood_SWISU',

       'Neighborhood_Sawyer', 'Neighborhood_SawyerW',

       'Neighborhood_Somerst', 'Neighborhood_StoneBr',

       'Neighborhood_Timber', 'Neighborhood_Veenker', 'Condition1_Feedr',

       'Condition1_Norm', 'Condition1_PosA', 'Condition1_PosN',

       'Condition1_RRAe', 'Condition1_RRAn', 'Condition1_RRNe',

       'Condition1_RRNn']
data_preprocessed = data_with_dummies[cols]
data_preprocessed
train_input = data_preprocessed.drop(['logPrice'],1)

train_target = data_preprocessed['logPrice']
scaler = StandardScaler()

scaler.fit(train_input)
train_scaled = scaler.transform(train_input)
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df_test
#Checking for duplicate values:

dup_data1 = df_test[df_test.duplicated()]

dup_data1.shape[0]
total = pd.isnull(df_test).sum().sort_values(ascending=False)

percentage = ((pd.isnull(df_test).sum() / pd.isnull(df_test).count()).sort_values(ascending=False))*100

no_values = pd.concat([total,percentage],axis = 1, keys = ['Total', 'Percent'])
no_values.head(30)
df_test1 = df_test.drop((no_values[no_values['Total'] > 1]).index,1)
df_test1.shape
factors_test = [ 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',

            'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
test_use = df_test1[factors_test].copy()
test_use
location1 = df_test1[['Neighborhood','Condition1']].copy()

data = df_test1[factors_test]

data_with_cat = data.join(location)

data_with_cat
pd.isnull(data_with_cat).sum()
data_with_cat[['GarageArea','GarageCars','TotalBsmtSF']].describe()
data_with_cat['GarageArea'].fillna(data_with_cat['GarageArea'].mean(),inplace=True)

data_with_cat['TotalBsmtSF'].fillna(data_with_cat['TotalBsmtSF'].mean(),inplace=True)

data_with_cat['GarageCars'].fillna(data_with_cat['GarageCars'].median(),inplace=True)
data_with_cat['Condition1'].fillna(data_with_cat['Condition1'].mode().values[0],inplace=True)

data_with_cat['Neighborhood'].fillna(data_with_cat['Neighborhood'].mode().values[0],inplace=True)
pd.isnull(data_with_cat).sum().max()
data_with_dummies = pd.get_dummies(data_with_cat, drop_first = True)
data_with_dummies
test_data = data_with_dummies.copy()
scaler = StandardScaler()

scaler.fit(test_data)
test_scaled = scaler.transform(test_data)
rf_reg = RandomForestRegressor()
max_features = ['auto','sqrt','log2']

n_estimators = [ int(x) for x in np.linspace(start=100,stop=1200,num=12)]

oob_score = ['True','False']

min_samples_leaf = [int(x) for x in np.linspace(start=1,stop=6,num=6)]

hyper_tune = { 'max_features' : max_features,

               'n_estimators' : n_estimators,

               'oob_score' : oob_score,

               'min_samples_leaf' : min_samples_leaf

    

}
rf_search = RandomizedSearchCV(estimator = rf_reg, param_distributions = hyper_tune, n_iter = 10, cv = 5, random_state = 1)
rf_search.fit(train_scaled,train_target)
rf_search.best_params_
rf_search.best_score_
rf_search.best_estimator_
test_pred = rf_search.predict(test_scaled)
output = pd.DataFrame({'ID' : df_test.Id, 'SalePrice' : np.exp(test_pred)})
output.to_csv('submission.csv',index=False)
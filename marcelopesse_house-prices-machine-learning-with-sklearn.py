import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plot data

import sklearn #scikit-learn library, where the magic happens!

import seaborn as sns # beautiful graphs

import re
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv',na_filter=False)

df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv',na_filter=False)
# Compute the correlation matrix

corr = df_train.corr()

# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap

cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, cmap=cmap, mask=mask, vmax=.3, center=0, annot=False, square=True, linewidths=.5, cbar_kws={"shrink": .5})
df_train[df_train.columns[1:]].corr()['SalePrice'][:].abs().sort_values(ascending=False)
df_train_20 = df_train[['Id','SalePrice','OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd']]

df_test_20 = df_test[['Id','OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd']]
df_train_20
print('df_train:')

print(df_train_20.isnull().sum())

print('\ndf_test:')

print(df_test_20.isnull().sum())
X_train = df_train_20.drop(columns=['SalePrice','Id'])

y_train = df_train_20['SalePrice']



X_test = df_test_20.drop(columns=['Id'])
X_test.fillna(0, inplace=True)
# from sklearn.ensemble import RandomForestClassifier



# tree = RandomForestClassifier(random_state=0)

# tree.fit(X_train, y_train)



# y_test = pd.Series(tree.predict(X_test))



# df_final = pd.concat([df_test['Id'], y_test], axis=1, sort=False)

# df_final = df_final.rename(columns={0:"SalePrice"})

# df_final.to_csv(r'020731.csv', index = False)
df_train['Neighborhood'].value_counts()
dict_neighbor = {

'NAmes'  :{'lat': 42.045830,'lon': -93.620767},

'CollgCr':{'lat': 42.018773,'lon': -93.685543},

'OldTown':{'lat': 42.030152,'lon': -93.614628},

'Edwards':{'lat': 42.021756,'lon': -93.670324},

'Somerst':{'lat': 42.050913,'lon': -93.644629},

'Gilbert':{'lat': 42.060214,'lon': -93.643179},

'NridgHt':{'lat': 42.060357,'lon': -93.655263},

'Sawyer' :{'lat': 42.034446,'lon': -93.666330},

'NWAmes' :{'lat': 42.049381,'lon': -93.634993},

'SawyerW':{'lat': 42.033494,'lon': -93.684085},

'BrkSide':{'lat': 42.032422,'lon': -93.626037},

'Crawfor':{'lat': 42.015189,'lon': -93.644250},

'Mitchel':{'lat': 41.990123,'lon': -93.600964},

'NoRidge':{'lat': 42.051748,'lon': -93.653524},

'Timber' :{'lat': 41.998656,'lon': -93.652534},

'IDOTRR' :{'lat': 42.022012,'lon': -93.622183},

'ClearCr':{'lat': 42.060021,'lon': -93.629193},

'StoneBr':{'lat': 42.060227,'lon': -93.633546},

'SWISU'  :{'lat': 42.022646,'lon': -93.644853}, 

'MeadowV':{'lat': 41.991846,'lon': -93.603460},

'Blmngtn':{'lat': 42.059811,'lon': -93.638990},

'BrDale' :{'lat': 42.052792,'lon': -93.628820},

'Veenker':{'lat': 42.040898,'lon': -93.651502},

'NPkVill':{'lat': 42.049912,'lon': -93.626546},

'Blueste':{'lat': 42.010098,'lon': -93.647269}

}
df_train['Lat'] = df_train['Neighborhood'].map(lambda neighbor: dict_neighbor[neighbor]['lat'])

df_train['Lon'] = df_train['Neighborhood'].map(lambda neighbor: dict_neighbor[neighbor]['lon'])



df_test['Lat'] = df_test['Neighborhood'].map(lambda neighbor: dict_neighbor[neighbor]['lat'])

df_test['Lon'] = df_test['Neighborhood'].map(lambda neighbor: dict_neighbor[neighbor]['lon'])
df_train.select_dtypes('object').columns
from sklearn import preprocessing



for columns in df_train.select_dtypes('object').columns:

    enc = preprocessing.LabelEncoder()

    enc.fit(pd.concat([df_train[columns].astype(str), df_test[columns].astype(str)],join='outer',sort=False))

    df_train[columns] = enc.transform(df_train[columns])

    df_test[columns] = enc.transform(df_test[columns])
# for col in df_train.columns:

#     print(df_train[col].astype(str).str.contains().any())
# X_train.fillna(0, inplace=True)

# X_test.fillna(0, inplace=True)
for columns in df_test.select_dtypes('object').columns:

    df_test[columns] = pd.to_numeric(df_test[columns],errors='coerce')

    

df_test.fillna(0, inplace=True)
# Compute the correlation matrix

corr = df_train.corr()



corr['SalePrice'].abs().sort_values(ascending=False).head(40)
# cols = ['SalePrice','OverallQual','GrLivArea','GarageCars','ExterQual','GarageArea','TotalBsmtSF','1stFlrSF','BsmtQual','KitchenQual','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd']

# g = sns.PairGrid(df_train[cols], height = 2.5)

# g.map(sns.scatterplot)
var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
df_train = df_train.drop((df_train[df_train['SalePrice']>700000]).index)

df_train = df_train.drop((df_train[df_train['TotalBsmtSF']>5000]).index)
var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
df_train = df_train.drop((df_train[df_train['GrLivArea']>4000]).index)
var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice');
from scipy.stats import norm

sns.distplot(df_train['SalePrice'], fit=norm);

fig = plt.figure()
df_train['SalePrice'] = np.log(df_train['SalePrice'])
sns.distplot(df_train['SalePrice'], fit=norm);

fig = plt.figure()
var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice');
df_train.isnull().sum().sum()
df_test.isnull().sum().sum()
df_train.dtypes[df_train.dtypes == 'float64']
df_test[['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageCars','GarageArea']] = df_test[['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageCars','GarageArea']].astype(int)



df_test.dtypes[df_test.dtypes == 'float64']
X_train = df_train.drop(columns=['SalePrice','Id','GarageArea','1stFlrSF','BsmtFinSF1','BsmtFinSF2'])

y_train = df_train['SalePrice']



X_test = df_test.drop(columns=['Id','GarageArea','1stFlrSF','BsmtFinSF1','BsmtFinSF2'])
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor





#tree = RandomForestClassifier(max_depth = 10, min_samples_split = 4, n_estimators = 500, random_state=0)

#tree = RandomForestClassifier(max_depth = 20, n_estimators = 1000, random_state=0)

#tree = RandomForestRegressor(max_depth = 20, n_estimators = 1000, random_state=0)

tree = RandomForestRegressor(max_depth = 20, n_estimators = 1000, random_state=0)





tree.fit(X_train, y_train)



y_test = pd.Series(tree.predict(X_test))



df_final = pd.concat([df_test['Id'], y_test], axis=1, sort=False)

df_final = df_final.rename(columns={0:"SalePrice"})

df_final['SalePrice'] = np.exp(df_final['SalePrice'])

df_final.to_csv(r'random_forest.csv', index = False)
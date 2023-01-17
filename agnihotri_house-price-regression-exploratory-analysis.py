import pandas as pd

import numpy as np

from scipy.stats import norm, probplot, linregress

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

import seaborn as sns

import os

import re
# Set the enviorment variables

train_file = os.path.join('..', 'input', 'house-prices-advanced-regression-techniques','train.csv')

train_file
df = pd.read_csv(train_file)

print("Shape of Data is {} rows and {} columns".format(df.shape[0], df.shape[1]))

df.sample(3)
# lambda function to calculate the null value distribution in dataframe

null_per = lambda x : ((x.isna()).sum()/len(x) * 100).sort_values(ascending=False)
null_per(df).head(20)
# Columns with more than 20% missing values

df[['PoolQC','MiscFeature','Alley','Fence','FireplaceQu']].sample(10)
print('What Kind of Data Types we are dealing with ?')

print("="*50)

df.dtypes.value_counts()
cat_cols = df.drop('Id', axis=1).dtypes[df.dtypes == 'object'].index

num_cols = df.drop('Id', axis=1).dtypes[(df.dtypes == 'int64') | (df.dtypes == 'float64')].index
temp = df.loc[:,cat_cols]

uni_count = dict()

print("Unique Values Distribution in Categorical Values")

print("="*50)

for col in temp.columns:

    uni_count[col] = len(temp[col].unique())

pd.Series(uni_count).sort_values()
print("Standard Deviation of Numerical Features")

print("="*50)

df.loc[:,num_cols].std().sort_values()
corr_df = df.loc[:,num_cols].corr()
fig,ax = plt.subplots(figsize=(12,10), dpi=100)

ax = sns.heatmap(corr_df, ax=ax, cmap='YlGnBu')
sale_price_corr = corr_df['SalePrice'].drop('SalePrice',axis=0).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(6,10),dpi=100)

ax =sns.barplot(x=sale_price_corr.values,y=sale_price_corr.keys(),)
corr_df_zoom = df[['SalePrice','OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF',

                   '1stFlrSF','FullBath','TotRmsAbvGrd']].corr()

fig,ax = plt.subplots(figsize=(4,3), dpi=100)

sns.heatmap(corr_df_zoom,ax=ax, cmap='Blues')
sns.pairplot(df[['SalePrice','OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF',

                   '1stFlrSF','FullBath','TotRmsAbvGrd']])
X = df['SalePrice']

fig, ax = plt.subplots(ncols=2,figsize=(16,6))

sns.distplot(X,ax=ax[0], fit=norm)

ax[0].title.set_text('Sale Price')

ax[0].grid()

probplot(X, fit=norm, plot=ax[1])

ax[1].title.set_text('Sale Price')

ax[1].grid()
X = np.log(df['SalePrice'])

fig, ax = plt.subplots(ncols=2,figsize=(16,6))

sns.distplot(X,ax=ax[0], fit=norm, color='green')

ax[0].title.set_text('Probability Mass Function')

ax[0].grid()



probplot(X, fit=norm, plot=ax[1])

ax[1].get_lines()[0].set_markerfacecolor('g')

ax[1].get_lines()[0].set_markeredgecolor('g')

ax[1].title.set_text('Q-Q Plot')

ax[1].grid()
X = df['GrLivArea']

fig, ax = plt.subplots(ncols=2,figsize=(16,6))

sns.distplot(X,ax=ax[0], fit=norm)

ax[0].title.set_text('Probability Mass Function')

ax[0].grid()



probplot(X, fit=norm, plot=ax[1])

ax[1].title.set_text('Q-Q Plot')

ax[1].grid()
X = np.log(df['GrLivArea'])

fig, ax = plt.subplots(ncols=2,figsize=(16,6))

sns.distplot(X,ax=ax[0], fit=norm)

ax[0].title.set_text('Probability Mass Function')

ax[0].grid()



probplot(X, fit=norm, plot=ax[1])

ax[1].get_lines()[0].set_markerfacecolor('g')

ax[1].get_lines()[0].set_markeredgecolor('g')

ax[1].title.set_text('Q-Q Plot')

ax[1].grid()
X = df['GarageArea']

fig, ax = plt.subplots(ncols=2,figsize=(16,6))

sns.distplot(X,ax=ax[0], fit=norm)

ax[0].title.set_text('Probability Mass Function')

ax[0].grid()



probplot(X, fit=norm, plot=ax[1])

ax[1].title.set_text('Q-Q Plot')

ax[1].grid()
X = df['TotalBsmtSF']

fig, ax = plt.subplots(ncols=2,figsize=(16,6))

sns.distplot(X,ax=ax[0], fit=norm)

ax[0].title.set_text('Probability Mass Function')

ax[0].grid()



probplot(X, fit=norm, plot=ax[1])

ax[1].title.set_text('Q-Q Plot')

ax[1].grid()
X = df[df['TotalBsmtSF'] < 4000]['TotalBsmtSF']

fig, ax = plt.subplots(ncols=2,figsize=(16,6))

sns.distplot(X,ax=ax[0], fit=norm)

ax[0].title.set_text('Probability Mass Function')

ax[0].grid()



probplot(X, fit=norm, plot=ax[1])

ax[1].title.set_text('Q-Q Plot')

ax[1].grid()
X = df[df['1stFlrSF'] < 3500]['1stFlrSF']

fig, ax = plt.subplots(ncols=2,figsize=(16,6))

sns.distplot(X,ax=ax[0], fit=norm)

ax[0].title.set_text('Probability Mass Function')

ax[0].grid()



probplot(X, fit=norm, plot=ax[1])

ax[1].title.set_text('Q-Q Plot')

ax[1].grid()
X = np.log(df[df['1stFlrSF'] < 3500]['1stFlrSF'])

fig, ax = plt.subplots(ncols=2,figsize=(16,6))

sns.distplot(X,ax=ax[0], fit=norm)

ax[0].title.set_text('Probability Mass Function')

ax[0].grid()



probplot(X, fit=norm, plot=ax[1])

ax[1].title.set_text('Q-Q Plot')

ax[1].grid()
X = df['TotRmsAbvGrd']

fig, ax = plt.subplots(ncols=2,figsize=(16,6))

sns.distplot(X,ax=ax[0], fit=norm)

ax[0].title.set_text('Probability Mass Function')

ax[0].grid()



probplot(X, fit=norm, plot=ax[1])

ax[1].title.set_text('Q-Q Plot')

ax[1].grid()
x = df['OverallQual']

y = df['SalePrice']

fig, ax = plt.subplots(figsize=(14,6))

sns.boxplot(x=x, y=y, ax=ax)

ax.grid()
x = np.log(np.array(df['GrLivArea']))

y = np.log(np.array(df['SalePrice']))

fig, ax = plt.subplots(figsize=(14,6))

sns.scatterplot(x=x, y=y, ax=ax)

lnrg = linregress((x,y))

y_lr = lnrg.slope*x + lnrg.intercept

sns.lineplot(x=x,y=y_lr,color='black')

ax.grid()
x = df['GarageCars']

y = np.log(df['SalePrice'])

fig, ax = plt.subplots(figsize=(14,6))

sns.boxplot(x=x, y=y, ax=ax)

ax.grid()
x = np.array(df['GarageArea'])

y = np.log(np.array(df['SalePrice']))

fig, ax = plt.subplots(figsize=(14,6))

sns.scatterplot(x=x, y=y, ax=ax)

lnrg = linregress((x,y))

y_lr = lnrg.slope*x + lnrg.intercept

sns.lineplot(x=x,y=y_lr,color='black')

ax.grid()
x = np.array(df['TotalBsmtSF'])

y = np.array(df['SalePrice'])

fig, ax = plt.subplots(figsize=(14,6))

sns.scatterplot(x=x, y=y, ax=ax)

lnrg = linregress((x,y))

y_lr = lnrg.slope*x + lnrg.intercept

sns.lineplot(x=x,y=y_lr,color='black')

ax.grid()
x = np.array(df['1stFlrSF'])

y = np.log(np.array(df['SalePrice']))

fig, ax = plt.subplots(figsize=(14,6))

sns.scatterplot(x=x, y=y, ax=ax)

lnrg = linregress((x,y))

y_lr = lnrg.slope*x + lnrg.intercept

sns.lineplot(x=x,y=y_lr,color='black')

ax.grid()
x = df['FullBath']

y = df['SalePrice']

fig, ax = plt.subplots(figsize=(14,6))

sns.boxplot(x=x, y=y, ax=ax)

ax.grid()
x = df['TotRmsAbvGrd']

y = df['SalePrice']

fig, ax = plt.subplots(figsize=(14,6))

sns.boxplot(x=x, y=y, ax=ax)

ax.grid()
x = np.array(df[df['TotalBsmtSF'] > 0]['GrLivArea'])

y = np.array(df[df['TotalBsmtSF'] > 0]['TotalBsmtSF'])

fig, ax = plt.subplots(figsize=(14,6))

sns.scatterplot(x=x, y=y, ax=ax)

lnrg = linregress((x,y))

y_lr = lnrg.slope*x + lnrg.intercept

sns.lineplot(x=x,y=y_lr,color='black')

ax.grid()
x = np.array(df['1stFlrSF'])

y = np.array(df['TotalBsmtSF'])

fig, ax = plt.subplots(figsize=(14,6))

sns.scatterplot(x=x, y=y, ax=ax)

lnrg = linregress((x,y))

y_lr = lnrg.slope*x + lnrg.intercept

sns.lineplot(x=x,y=y_lr,color='black')

ax.grid()
df['has_basement'] = df['TotalBsmtSF'].apply(lambda x : 'Yes' if x > 0 else 'No')

df['has_garage'] = df['GarageArea'].apply(lambda x : 'Yes' if x > 0 else 'No')
print('Properties Having Basement-Distribution : ')

print(df['has_basement'].value_counts()/len(df)*100)

print('='*50)

print('Properties Having Garage-Distribution : ')

print(df['has_garage'].value_counts()/len(df)*100)

print('='*50)
df = df.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'],axis=1)
cat_cols = df.dtypes[df.dtypes == 'object'].index

cat_cols
X_dum = pd.get_dummies(df.loc[:,df.dtypes[df.dtypes == 'object'].index])

y = np.log(df['SalePrice'])
clf = RandomForestRegressor(n_estimators=100)

clf.fit(X_dum, y)
feat_import = pd.Series(dict(zip(X_dum.columns,clf.feature_importances_))).sort_values(ascending=False)

dummy_cols = list(X_dum.columns)

rows = dict()

for col in cat_cols:

    cols_to_sum = [w for w  in dummy_cols if re.search(col,w) != None]

    rows[col] = np.median(feat_import[cols_to_sum].sum())

rows = pd.Series(rows).sort_values(ascending=False)

print("Sum of Feature Importance of each feature, across it different categories")

print('='*80)

rows.head(10)
feat_import = pd.Series(dict(zip(X_dum.columns,clf.feature_importances_))).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10,12),dpi=100)

x = list(rows.values)

y = list(rows.index)

sns.barplot(x=x, y=y)
x = df['ExterQual'].sort_values()

y = df['SalePrice']

fig, ax = plt.subplots(ncols=2,figsize=(12,4),dpi=100)

sns.boxplot(x=x, y=y, ax=ax[1])

x = x.value_counts().sort_index()

sns.barplot(x=x.keys(),y=x.values,ax=ax[0])

ax[1].grid()
x = df['BsmtQual'].sort_values()

y = df['SalePrice']

fig, ax = plt.subplots(ncols=2,figsize=(12,4),dpi=100)

sns.boxplot(x=x, y=y, ax=ax[1])

x = x.value_counts().sort_index()

sns.barplot(x=x.keys(),y=x.values,ax=ax[0])

ax[1].grid()
x = df['Neighborhood'].sort_values()

y = df['SalePrice']

fig, ax = plt.subplots(ncols=2,figsize=(16,4),dpi=100)

sns.boxplot(x=x, y=y, ax=ax[1])

x = x.value_counts().sort_index()

sns.barplot(x=x.keys(),y=x.values,ax=ax[0])

ax[1].grid()
x = df['GarageType'].sort_values()

y = df['SalePrice']

fig, ax = plt.subplots(ncols=2,figsize=(16,4),dpi=100)

sns.boxplot(x=x, y=y, ax=ax[1])

x = x.value_counts().sort_index()

sns.barplot(x=x.keys(),y=x.values,ax=ax[0])

ax[1].grid()
x = df['KitchenQual'].sort_values()

y = df['SalePrice']

fig, ax = plt.subplots(ncols=2,figsize=(16,4),dpi=100)

sns.boxplot(x=x, y=y, ax=ax[1])

x = x.value_counts().sort_index()

sns.barplot(x=x.keys(),y=x.values,ax=ax[0])

ax[1].grid()
x = df['MSZoning'].sort_values()

y = df['SalePrice']

fig, ax = plt.subplots(ncols=2,figsize=(16,4),dpi=100)

sns.boxplot(x=x, y=y, ax=ax[1])

x = x.value_counts().sort_index()

sns.barplot(x=x.keys(),y=x.values,ax=ax[0])

ax[1].grid()
num_cols_selected = ['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath',

                 'TotRmsAbvGrd']

cat_cols_selected = ['ExterQual', 'BsmtQual', 'Neighborhood', 'GarageType', 'KitchenQual','MSZoning']

target = 'SalePrice'
train, test = train_test_split(df, random_state=42, test_size=0.2)
train[cat_cols_selected] = train[cat_cols_selected].fillna('missing')

test[cat_cols_selected] = test[cat_cols_selected].fillna('missing')



one_hot = OneHotEncoder()

one_hot.fit(train[cat_cols_selected])

X_train_cat = one_hot.transform(train[cat_cols_selected]).toarray()

X_test_cat = one_hot.transform(test[cat_cols_selected]).toarray()
ss = StandardScaler()

ss.fit(train[num_cols_selected])

X_train_num = ss.transform(train[num_cols_selected])

X_test_num = ss.transform(test[num_cols_selected])
X_train = np.c_[X_train_cat, X_train_num]

X_test = np.c_[X_test_cat, X_test_num]



y_train, y_test = train[target], test[target]
clf = RandomForestRegressor(n_estimators=100, bootstrap=True)

clf.fit(X_train, np.log(y_train))
y_pred = clf.predict(X_test)

mse = mean_squared_error(np.log(y_test), y_pred)

rmse = np.sqrt(mse)

print('Root Mean Squared Error - Base Model - {}'.format(rmse))
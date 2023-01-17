# NumPy for numerical computing

import numpy as np



# Pandas for DataFrames

import pandas as pd

pd.set_option('display.max_columns', 100)



# Matplotlib for visualization

from matplotlib import pyplot as plt

# display plots in the notebook

%matplotlib inline 



# Seaborn for easier visualization

import seaborn as sns
# Load AMES Housing training and test data from CSV

hs_train = pd.read_csv('../input/train.csv')

hs_test = pd.read_csv('../input/test.csv')
# Dataframe dimensions

hs_train.shape
# Column data types

hs_train.dtypes.reset_index().T
miss_train = hs_train.isnull().sum().sort_values(ascending=False)
train_mask = (miss_train)/1460 > 0.15

miss_train[train_mask].index
missing_list=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu','LotFrontage']

hs_train = hs_train.drop(missing_list, axis=1)

hs_test = hs_test.drop(missing_list, axis=1)
# Fill missing numerical values

for column in hs_train.select_dtypes(exclude=['object']):

    hs_train[column] = hs_train[column].fillna(0)

    

for column in hs_test.select_dtypes(exclude=['object']):

    hs_test[column] = hs_train[column].fillna(0)
# Fill missing categorical values

for column in hs_train.select_dtypes(include=['object']):

    hs_train[column] = hs_train[column].fillna('missing')
# Plot histogram grid

plt.figure()

hs_train.hist(layout=(8,5),color='k', alpha=0.5, bins=50,figsize=(20,30));
plt.figure(figsize=(14,3))

sns.boxplot(x='SalePrice', data=hs_train, color='mediumspringgreen')

plt.xlim(0,800000)

plt.figure(figsize=(14,3))

sns.violinplot(x='SalePrice', data=hs_train, color='mediumspringgreen')

plt.xlim(0,800000)
# Plot bar plot for each categorical feature

fig, axes =plt.subplots(10,4, figsize=(20,40), sharex=True)

axes = axes.flatten()

object_bol = hs_train.dtypes == 'object'

for ax, catplot in zip(axes, hs_train.dtypes[object_bol].index):

    sns.countplot(y=catplot, data=hs_train, ax=ax)



plt.tight_layout()  

plt.show()
sns.set(style="white")

corr = hs_train.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(220, 10, as_cmap=True)

plt.figure(figsize=(35,20))

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, annot=True, fmt='.1f',

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, cbar=False)
k = 10 

cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(hs_train[cols].values.T)

sns.set(font_scale=1.25)

sns.heatmap(cm, annot=True, square=True, fmt='.2f', cmap=cmap,

                 annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
corr.nlargest(k, 'SalePrice')['SalePrice'].index
col = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars',

       'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
sns.pairplot(hs_train[col], size = 3)
hs_train.plot.scatter(x='GrLivArea',y='SalePrice')
grbol = hs_train['GrLivArea'] < 4500

hs_train = hs_train[grbol]
from sklearn.model_selection import KFold, cross_val_score

from sklearn.linear_model import LinearRegression



X = hs_train[['OverallQual']].values

y = hs_train.SalePrice.values



kfold = KFold(n_splits=5, random_state=4, shuffle=True)

model = LinearRegression()

results = cross_val_score(model, X, y, cv=kfold, scoring='r2')
results.mean()
from sklearn.ensemble import RandomForestRegressor
hs_train_simp_bin = pd.get_dummies(hs_train)



y = hs_train_simp_bin.pop('SalePrice')

X = hs_train_simp_bin.values



kfold = KFold(n_splits=5, random_state=4, shuffle=True)

model = RandomForestRegressor()

results = cross_val_score(model, X, y, cv=kfold, scoring='r2')
results.mean()
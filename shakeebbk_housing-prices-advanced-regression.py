import pandas as pd

import numpy as np



from scipy.stats import norm

from scipy import stats



import matplotlib.pyplot as plt

import seaborn as sns
train_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

train_df.head()
test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

test_df.head()
print(f'Shape of train data {train_df.shape}')

print(f'Shape of test data {test_df.shape}')
y_train = train_df['SalePrice']
X_train = train_df.drop('SalePrice', axis=1)
print(f'Shape of train data {X_train.shape}')

print(f'Shape of test data {test_df.shape}')
master_df = pd.concat([X_train, test_df], axis='rows')

master_df.shape
# would need Id in submission

submission = test_df['Id']



master_df.drop('Id', axis=1, inplace=True)

master_df.shape
# Understanding and visualization

corr = master_df.corr()

plt.figure(figsize=(12, 10))

sns.heatmap(corr, vmax=.8, vmin=-0.8, square=True)

plt.show()
# handle columns with high collinearity

corr_unstacked = corr.abs().unstack()

pairs = corr_unstacked[(corr_unstacked > 0.8) & (corr_unstacked < 1)]

pairs
# we will look at thier correlation with price and drop the ones which are less correlated

sns.heatmap(train_df[['SalePrice', 'YearBuilt', 'GarageYrBlt']].corr(), annot=True, square=True)

plt.show()



sns.heatmap(train_df[['SalePrice', 'TotalBsmtSF', '1stFlrSF']].corr(), annot=True, square=True)

plt.show()



sns.heatmap(train_df[['SalePrice', 'GarageCars', 'GarageArea']].corr(), annot=True, square=True)

plt.show()



sns.heatmap(train_df[['SalePrice', 'TotRmsAbvGrd', 'GrLivArea']].corr(), annot=True, square=True)

plt.show()
# drop GarageYrBlt, 1stFlrSF, GarageCars and TotRmsAbvGrd

master_df.drop(columns=['GarageYrBlt', '1stFlrSF', 'GarageCars', 'TotRmsAbvGrd'], inplace=True)

master_df.shape
def print_missing(df):

    cols = list(df.columns[(df.isnull().sum() / df.shape[0]) > 0])

    print((df[cols].isnull().sum() / df.shape[0]).sort_values(ascending=False))

    

print_missing(master_df)
# Consider SaleType

print(master_df['SaleType'].describe())



plt.figure(figsize=(12, 8))

sns.boxplot(x='SaleType', y='SalePrice', data=train_df)

plt.show()



# fill with most freq

master_df['SaleType'].fillna(master_df['SaleType'].mode()[0], inplace=True)
print_missing(master_df)
# Consider PoolQC

# lets see if its related to pool area

sns.boxplot(x='PoolQC', y='PoolArea', data=train_df)

plt.show()



sns.boxplot(x='PoolQC', y='SalePrice', data=train_df)

plt.show()



sns.scatterplot(x='SalePrice', y='PoolArea', data=train_df)

plt.show()
# we can simply say NA - NoPool

master_df['PoolQC'].fillna('NoPool', inplace=True)

print_missing(master_df)
master_df['MiscFeature'].fillna('None', inplace=True)

print_missing(master_df)
master_df['Alley'].fillna('NoAlley', inplace=True)

print_missing(master_df)
# replace NA with none

cols = ['Fence', 'FireplaceQu', 'GarageCond', 'GarageFinish',

           'GarageType', 'BsmtCond', 'BsmtExposure', 'BsmtQual',

           'BsmtFinType2', 'BsmtFinType1', 'MasVnrType', 'GarageQual']

for c in cols:

    master_df[c].fillna('None', inplace=True)

print_missing(master_df)
print(master_df['LotFrontage'].describe())



# replace with median

master_df['LotFrontage'].fillna(master_df['LotFrontage'].median(), inplace=True)
print(master_df['MasVnrArea'].describe())

# replace missing values with zero

master_df['MasVnrArea'].fillna(0, inplace=True)
# replace NA with mode

cols = ['Functional', 'MSZoning', 'Utilities',

       'KitchenQual', 'Electrical', 'Exterior2nd', 'Exterior1st']

for c in cols:

    master_df[c].fillna(master_df[c].mode()[0], inplace=True)



print_missing(master_df)
# replace wuith zeors

cols = ['BsmtHalfBath', 'BsmtFullBath', 'GarageArea',

       'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF2', 'BsmtFinSF1']

for c in cols:

    master_df[c].fillna(0, inplace=True)

    

print_missing(master_df)
# add dummies for categorical vars

master_df = pd.get_dummies(master_df, drop_first=True)

master_df.shape
# plot y_train

sns.distplot(y_train, fit=norm)

plt.show()



stats.probplot(y_train, plot=plt)

plt.show()



# since it is right skewed, log transformation should help

y_train = np.log1p(y_train)



sns.distplot(y_train, fit=norm)

plt.show()



stats.probplot(y_train, plot=plt)

plt.show()
# plot y_train

sns.distplot(master_df['GrLivArea'], fit=norm)

plt.show()



stats.probplot(master_df['GrLivArea'], plot=plt)

plt.show()



# since it is right skewed, log transformation should help

master_df['GrLivArea'] = np.log1p(master_df['GrLivArea'])



sns.distplot(master_df['GrLivArea'], fit=norm)

plt.show()



stats.probplot(master_df['GrLivArea'], plot=plt)

plt.show()
X_train = master_df[:1460]

X_test = master_df[1460:]



print(X_train.shape)

print(X_test.shape)
from IPython.core.display import display, HTML

HTML('''<script> </script> <form action="javascript:IPython.notebook.execute_cells_above()"><input type="submit" id="toggleButton" value="Run all above Cells"></form>''')
y_train = pd.DataFrame(y_train)
y_train.shape
y_train.describe()
from sklearn.preprocessing import StandardScaler



xscaler = StandardScaler()

# yscaler = StandardScaler()



cols = X_train.columns

X_train = pd.DataFrame(xscaler.fit_transform(X_train), columns=cols)

# y_train = pd.DataFrame(yscaler.fit_transform(y_train))
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso



param = {'alpha': [0.0001, 0.001, 0.005, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9, 1]}

model = GridSearchCV(estimator=Lasso(),

                     param_grid=param,

                     cv=5,

                     scoring='r2')

model.fit(X_train, y_train)
print(model.best_score_)
from sklearn.metrics import mean_squared_error

print(f'RMSE: {np.sqrt(mean_squared_error(model.predict(X_train), y_train))}')
print(model.best_params_)
y_train.head()
np.expm1(y_train).head()
X_test = xscaler.transform(X_test)

y_pred = model.predict(X_test)

y_pred[0:10]
# y_pred_inv = yscaler.inverse_transform(y_pred)

# y_pred_inv[0:10]
y_pred_inv = np.expm1(y_pred)

y_pred_inv[0:10]
submission_df = test_df[['Id']]

submission_df['SalePrice'] = y_pred_inv

print(submission_df.shape)

submission_df.head()
submission_df.to_csv('submission.csv', index=False)
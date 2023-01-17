import numpy as np 

import pandas as pd 



import random as rnd

%matplotlib inline

import pandas as pd

pd.options.display.max_columns = 100

from matplotlib import pyplot as plt

import matplotlib

matplotlib.style.use('ggplot')



pd.options.display.max_rows = 100

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train.describe()
test.describe()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline 
corr = train.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()

plt.figure(figsize=(12, 12))

sns.heatmap(corr, vmax=1, square=True)
k = 20 #number of variables for heatmap

cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

plt.figure(figsize=(12, 12))

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
fig, ax = plt.subplots()

ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
from scipy.stats import norm, skew

from scipy import stats

sns.distplot(train['SalePrice'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train['SalePrice'])

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
train["SalePrice"] = np.log1p(train["SalePrice"])
sns.distplot(train['SalePrice'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train['SalePrice'])

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
ntrain = train.shape[0]

ntest = test.shape[0]



# get the targets

y_train_sale = train.SalePrice.values



# combine train and test

combined = pd.concat((train, test)).reset_index(drop=True)

combined.drop(['SalePrice'], axis=1, inplace=True)
all_data_na = (combined.isnull().sum() / len(combined)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data
f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='90')

sns.barplot(x=all_data_na.index, y=all_data_na)

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)
combined['MasVnrArea'] = combined['MasVnrArea'].fillna(0.0)

combined["MasVnrType"] = combined["MasVnrType"].fillna("None")

combined['LotFrontage'] = combined['LotFrontage'].fillna(combined['LotFrontage'].median())

combined['BsmtFinSF1'] = combined['BsmtFinSF1'].fillna(0.0)

combined['BsmtFinSF2'] = combined['BsmtFinSF2'].fillna(0.0)

combined['BsmtUnfSF'] = combined['BsmtUnfSF'].fillna(0.0)

combined['TotalBsmtSF'] = combined['TotalBsmtSF'].fillna(0.0)

combined['BsmtFullBath'] = combined['BsmtFullBath'].fillna(0)

combined['BsmtHalfBath'] = combined['BsmtHalfBath'].fillna(0)

combined['GarageYrBlt'] = combined['GarageYrBlt'].fillna(0)

combined['GarageCars'] = combined['GarageCars'].fillna(0)

combined['GarageArea'] = combined['GarageArea'].fillna(0)

combined['GarageFinish'] = combined['GarageFinish'].fillna('None')



# using the most frequent zone

combined['MSZoning'] = combined['MSZoning'].fillna(combined['MSZoning'].mode()[0])



combined = combined.drop(['Utilities'], axis=1)



# most common functionality

combined["Functional"] = combined["Functional"].fillna("Typ")



combined['Electrical'] = combined['Electrical'].fillna(combined['Electrical'].mode()[0])

combined['KitchenQual'] = combined['KitchenQual'].fillna(combined['KitchenQual'].mode()[0])

combined['Exterior1st'] = combined['Exterior1st'].fillna(combined['Exterior1st'].mode()[0])

combined['Exterior2nd'] = combined['Exterior2nd'].fillna(combined['Exterior2nd'].mode()[0])

combined['SaleType'] = combined['SaleType'].fillna(combined['SaleType'].mode()[0])

combined['MSSubClass'] = combined['MSSubClass'].fillna("None")

combined['PoolQC'] = combined['PoolQC'].fillna('None')

combined['MiscFeature'] = combined['MiscFeature'].fillna('None')

combined['Alley'] = combined['Alley'].fillna('None')

combined['Fence'] = combined['Fence'].fillna('None')

combined['FireplaceQu'] = combined['FireplaceQu'].fillna('None')

combined["LotFrontage"] = combined.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    combined[col] = combined[col].fillna('None')



for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    combined[col] = combined[col].fillna('None')
from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')



for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(combined[c].values)) 

    combined[c] = lbl.transform(list(combined[c].values))
combined['TotalSF'] = combined['TotalBsmtSF'] + combined['1stFlrSF'] + combined['2ndFlrSF']
combined = pd.get_dummies(combined)
train = combined[:ntrain]

test = combined[ntrain:]
from sklearn.model_selection import train_test_split, cross_val_score
x_train, x_test, y_train, y_test = train_test_split(train, y_train_sale, test_size=0.1, random_state=200)
from sklearn import ensemble, tree, linear_model

from sklearn.metrics import r2_score, mean_squared_error
def get_score(prediction, lables):    

    print('R2: {}'.format(r2_score(prediction, lables)))

    print('RMSE: {}'.format(np.sqrt(mean_squared_error(prediction, lables))))



def train_test(estimator, x_trn, x_tst, y_trn, y_tst):

    prediction_train = estimator.predict(x_trn)

    

    get_score(prediction_train, y_trn)

    prediction_test = estimator.predict(x_tst)

    

    get_score(prediction_test, y_tst)
GBR = ensemble.GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=3, max_features='sqrt',

                                               min_samples_leaf=15, min_samples_split=10, loss='huber').fit(x_train, y_train)

train_test(GBR, x_train, x_test, y_train, y_test)
model = GBR.fit(train, y_train_sale)

output = np.expm1(model.predict(test))
pd.DataFrame({'Id': test.Id, 'SalePrice': output}).to_csv('output.csv', index =False)
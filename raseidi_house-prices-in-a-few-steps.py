import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_selection import VarianceThreshold

plt.style.use('ggplot')
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

train['Dataset'] = 'train'

test['Dataset'] = 'test'



y = train.SalePrice

train.drop('SalePrice', axis=1, inplace=True)



df = pd.concat([train, test])

df.set_index(['Id', 'Dataset'], inplace=True)
missing = df.isnull().sum()

missing[missing > 0].index

df.drop(missing[missing > 0].index, axis=1, inplace=True)
df.columns[df.isna().any()]
categorical = [f for f in df.columns if df.dtypes[f] == 'object']



for c in categorical:

    df['cat_' + c] = LabelEncoder().fit_transform(df[c])

    

df.drop(categorical, axis=1, inplace=True)



print(df.head())
from sklearn.feature_selection import VarianceThreshold



selector = VarianceThreshold(0.15)

selector.fit(df)

# get support returns True for columns with var > 0.15

print('Columns with variance lower than 0.15: ', df.columns[~selector.get_support()])

df.drop(df.columns[~selector.get_support()], axis=1, inplace=True)
def boxplot(x, y, **kwargs):

    sns.boxplot(x=x, y=y)

    x=plt.xticks(rotation=65)



categorical = [c for c in df.columns if c.startswith('cat_')]

tmp = df[df.index.get_level_values(1) == 'train'].copy()

tmp.loc[:, 'SalePrice'] = y.values

f = pd.melt(tmp, id_vars=['SalePrice'], value_vars=categorical)

plt.figure(dpi=100, figsize=(16,8))

g = sns.FacetGrid(f, col="variable",  col_wrap=4, sharex=False, sharey=False)

g = g.map(boxplot, "value", "SalePrice")

plt.show()
to_binary = ['cat_PavedDrive', 'cat_HeatingQC', 'cat_HouseStyle', 'cat_LotConfig',

             'cat_LotShape', 'cat_RoofMatl', 'cat_RoofStyle']

one_hot_encoding = ['cat_BldgType', 'cat_ExterQual', 'cat_ExterCond',

                    'cat_SaleCondition', 'cat_Condition1', 'cat_LandContour',

                    'cat_Foundation']



df.loc[:, 'cat_PavedDrive'] = np.where(

    df.loc[:, 'cat_PavedDrive'].isin([0, 1]), 0, 1

)



df.loc[:, 'cat_HeatingQC'] = np.where(

    df.loc[:, 'cat_HeatingQC'].isin([1, 2, 3, 4]), 0, 1

)



df.loc[:, 'cat_HouseStyle'] = np.where(

    df.loc[:, 'cat_HouseStyle'].isin([0,1,4,6,7]), 0, 1

)



df.loc[:, 'cat_LotConfig'] = np.where(

    df.loc[:, 'cat_LotConfig'].isin([1, 2, 4]), 0, 1

)



df.loc[:, 'cat_LotShape'] = np.where(

    df.loc[:, 'cat_LotShape'].isin([0,1,3]), 0, 1

)



df.loc[:, 'cat_RoofStyle'] = np.where(

    df.loc[:, 'cat_RoofStyle'].isin([1,2,4,5]), 0, 1

)



df.loc[:, 'cat_RoofMatl'] = np.where(

    df.loc[:, 'cat_RoofMatl'].isin([0,2,3,4,5,6]), 0, 1

)



selector = VarianceThreshold(0.1)

selector.fit(df)

df.drop(df.columns[~selector.get_support()], axis=1, inplace=True)



df.loc[:, 'cat_BldgType'] = df.loc[:, 'cat_BldgType'].map({

    1: 0, 2: 0, 3: 0,

    4: 1,

    0: 2

})



df.loc[:, 'cat_ExterCond'] = df.loc[:, 'cat_ExterCond'].map({

    1: 0, 3: 0,

    0: 1,

    2: 2, 4: 2

})



df.loc[:, 'cat_SaleCondition'] = df.loc[:, 'cat_SaleCondition'].map({

    0: 0, 1: 0, 2: 0, 3: 0,

    4: 1,

    5: 2

})



df.loc[:, 'cat_Condition1'] = df.loc[:, 'cat_Condition1'].map({

    1: 0, 5: 0, 6: 0,

    0: 1, 3: 1, 4: 1, 6: 1, 7: 1, 8: 1,

    2: 2

})



df.loc[:, 'cat_LandContour'] = df.loc[:, 'cat_LandContour'].map({

    0: 0,

    1: 1, 2: 1,

    3: 2

})



df.loc[:, 'cat_Foundation'] = df.loc[:, 'cat_Foundation'].map({

    0: 0, 1: 0,

    3: 1, 4: 1, 5: 1,

    2: 2

})



df = pd.get_dummies(df, columns=one_hot_encoding)



print('Columns binirized: {}\nColumns transformed by One Hot Enconding technique: {}'.format(to_binary, one_hot_encoding))
tmp = df[df.index.get_level_values(1) == 'train'].copy()

tmp.loc[:, 'SalePrice'] = y.values

tmp.loc[:, 'YearBuilt'] = pd.cut(tmp.YearBuilt, 7, labels=range(7)).astype('int')

sns.swarmplot(x='YearBuilt', y='SalePrice', hue='YearBuilt',

              data=tmp)

plt.show()



df.GrLivArea = df.GrLivArea.apply(np.log)

df.loc[:, 'OverallQual'] = pd.cut(df.OverallQual, 3, labels=[0, 1, 2])

df = pd.get_dummies(df, columns=['OverallQual'])

df.loc[:, 'TotRmsAbvGrd'] = pd.cut(df.TotRmsAbvGrd, 4, labels=[0, 1, 2, 3])

df.loc[:, 'YearRemodAdd'] = pd.cut(df.YearRemodAdd, 3, labels=[0, 1, 2]).astype('int')

df = pd.get_dummies(df, columns=['TotRmsAbvGrd'])

df.loc[:, '1stFlrSF'] = df['1stFlrSF'].apply(lambda x: np.log(x) if x else x)

df.loc[:, '2ndFlrSF'] = df['2ndFlrSF'].apply(lambda x: np.log(x) if x else x)

# Analyzing YearBuilt we noticed that values > 1982 are very

# higher than other. Thus, we will transform this column

# into binary

# train_.loc[:, 'Year_tmp'] = pd.cut(train_[var], 10) #, labels=range(10))

df.loc[:, 'YearBuilt'] = np.where(

    df.loc[:, 'YearBuilt'] > 1982, 1, 0

)

# From here, Neighborhood column looks really useless. Drop it.

df.drop(['MSSubClass', 'OpenPorchSF', 'BedroomAbvGr',

         'LotArea', 'YrSold'], axis=1, inplace=True)
df.reset_index(inplace=True)

train = df[df.Dataset == 'train'].copy()

train['SalePrice'] = y.copy()

test = df[df.Dataset == 'test'].copy()



train.drop('Dataset', axis=1, inplace=True)

train.set_index('Id', inplace=True)

test.drop('Dataset', axis=1, inplace=True)

test.set_index('Id', inplace=True)



train.loc[:, 'SalePrice'] = train.SalePrice.apply(np.log)

X = train.drop('SalePrice', axis=1)

y = train.SalePrice
from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor(n_estimators=100)

reg.fit(X, y)



tmp = pd.DataFrame({

    'Feature': X.columns,

    'Importance': reg.feature_importances_

})

tmp.sort_values(by='Importance', ascending=False).head()
from sklearn.ensemble import GradientBoostingRegressor

gb_bestparams = {

    'loss': 'huber',

    'max_features': 'log2',

    'min_samples_split': 10,

    'n_estimators': 500

}

reg = GradientBoostingRegressor(**gb_bestparams)

reg.fit(X, y)



test_ = test.copy()

test_['SalePrice'] = reg.predict(test)

test_['SalePrice'] = test_['SalePrice'].apply(lambda x: np.floor(np.exp(x))) # to submit the predictions we need to transform it back to the original scale.

test_.reset_index(inplace=True)

test_[['Id', 'SalePrice']].to_csv('submission.csv', index=False)

test_[['Id', 'SalePrice']].head()

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import scipy as sp

from sklearn.metrics import make_scorer

from sklearn.metrics import mean_squared_error as mse

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score
train = pd.read_csv('../input/2019-2nd-ml-month-with-kakr/train.csv')

test = pd.read_csv('../input/2019-2nd-ml-month-with-kakr/test.csv')
print('train shape : ',train.shape)

print('test shape : ',test.shape)
train.head()
for df in [train, test]:

    df['year'] = df['date'].apply(lambda x : x[0:4], range(len(df)))

    df['month'] = df['date'].apply(lambda x : x[4:6], range(len(df)))

    df['day'] = df['date'].apply(lambda x : x[6:8], range(len(df)))
train.isnull().sum()
test.isnull().sum()
train.columns
train['year'].value_counts()
test['year'].value_counts()
sns.distplot(train['price'])
train['price'] = sp.special.log1p(train['price'])
sns.distplot(train['price'])
sns.boxplot(train['bedrooms'], train['price'])
sns.boxplot(train['bathrooms'], train['price'])
sns.regplot(train['sqft_living'], train['price'])
sns.regplot(train['sqft_living15'], train['price'])
sns.regplot(train['sqft_living']-train['sqft_living15'], train['price'])
sns.regplot(train['sqft_lot'], train['price'])
sns.regplot(train['sqft_lot15'], train['price'])
sns.regplot(train['sqft_lot'] - train['sqft_lot15'], train['price'])
sns.boxplot(train['floors'], train['price'])
sns.boxplot(train['waterfront'], train['price'])
sns.boxplot(train['view'], train['price'])
sns.boxplot(train['condition'], train['price'])
sns.boxplot(train['grade'], train['price'])
sns.regplot(train['sqft_above'], train['price'])
sns.regplot(train['sqft_basement'], train['price'])
sns.regplot(train['yr_built'], train['price'])
train.columns
sns.regplot(train['sqft_living'] + train['sqft_lot'], train['price'])
sns.regplot(train['sqft_living'] + train['sqft_lot'] + train['sqft_living15'], train['price'])
corr = train.corr(method = 'spearman')

corr = corr.nlargest(n = 10, columns = 'price')

plt.figure(figsize = (14, 10))

sns.heatmap(corr, fmt = '.2f', annot = True)

sns.set(font_scale = 1.25)
sns.boxplot(train['grade'], train['price'])
train[train['grade'] >= 13]
test[test['grade'] >= 13]
sns.regplot(train['sqft_living'], train['price'])
train[train['sqft_living'] >= 8000]
test[test['sqft_living'] >= 8000]
sns.regplot(train['sqft_living15'], train['price'])
train[train['sqft_living15'] >= 6000]
test[test['sqft_living15'] >= 6000]
sns.regplot(train['sqft_above'], train['price'])
train[train['sqft_above'] >= 7000]
test[test['sqft_above'] >= 7000]
sns.boxplot(train['bathrooms'], train['price'])
sns.distplot(train['bathrooms'])
sns.distplot(test['bathrooms'])
train[train['bathrooms'] > 6]
test[test['bathrooms'] > 6]
test_id = test.id
for df in [train, test]:

    df.drop(columns = ['id', 'date', 'lat', 'long'], inplace = True)
K_fold = KFold(n_splits = 10, shuffle = True, random_state = 42)
X_train = train.drop(columns = 'price')

y_train = train.price
clf = RandomForestRegressor(n_estimators = 15, n_jobs = -1)
scoring = make_scorer(mse)

score = cross_val_score(clf, X_train, y_train, cv = K_fold, n_jobs = 1, scoring = scoring)
print(sp.special.expm1(score) ** 0.5)
clf.fit(X_train, y_train)
prediction = clf.predict(test)
prediction = sp.special.expm1(prediction)
prediction
submission = pd.DataFrame({'id' : test_id, 'price' : prediction})
submission
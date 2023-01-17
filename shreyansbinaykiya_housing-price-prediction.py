import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns 

from scipy import stats

from scipy.stats import norm, skew

from sklearn.metrics import r2_score

import os

print(os.listdir("../input"))

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
corr = train.corr()
corr[corr['SalePrice']>0.3].index
train = train[['LotFrontage', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',

       'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',

       'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',

       'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'SalePrice']]

test=test[['LotFrontage', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',

       'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',

       'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',

       'GarageArea', 'WoodDeckSF', 'OpenPorchSF']]
#missing data

total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(25)
#dealing with missing data

train = train.drop((missing_data[missing_data['Total'] > 81]).index,1)
train.isnull().sum().sort_values(ascending=False).head(20)
#missing data

total_test = test.isnull().sum().sort_values(ascending=False)

percent_test = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total_test, percent_test], axis=1, keys=['Total', 'Percent'])

missing_data.head(25)
#dealing with missing data

test = test.drop((missing_data[missing_data['Total'] > 78]).index,1)
test.isnull().sum().sort_values(ascending=False).head(20)
train.isnull().sum().sort_values(ascending = False).head(20)
# Categorical boolean mask

categorical_feature_mask = train.dtypes==object

# filter categorical columns using mask and turn it into alist

categorical_cols = train.columns[categorical_feature_mask].tolist()
categorical_cols
#data = pd.get_dummies(data, columns=categorical_cols)
#from sklearn.preprocessing import LabelEncoder

#labelencoder = LabelEncoder()

#train[categorical_cols] = train[categorical_cols].apply(lambda col: labelencoder.fit_transform(col.astype(str)))
# Categorical boolean mask

categorical_feature_mask_test = test.dtypes==object

# filter categorical columns using mask and turn it into alist

categorical_cols_test = test.columns[categorical_feature_mask_test].tolist()
#from sklearn.preprocessing import LabelEncoder

#labelencoder = LabelEncoder()

#test[categorical_cols_test] = test[categorical_cols_test].apply(lambda col: labelencoder.fit_transform(col.astype(str)))
train.isnull().sum().sort_values(ascending=False).head(20)
test.isnull().sum().sort_values(ascending=False).head(20)
train['GarageYrBlt'] = train['GarageYrBlt'].fillna(train['GarageYrBlt'].mean())

train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].mean())
k = 15 #number of variables for heatmap

plt.figure(figsize=(16,8))

corrmat = train.corr()

# picking the top 15 correlated features

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()

train = train[cols]
cols
test=test[cols.drop('SalePrice')]
test.isnull().sum().sort_values(ascending=False).head(20)
test['GarageYrBlt'] = test['GarageYrBlt'].fillna(test['GarageYrBlt'].mean())

test['MasVnrArea'] = test['MasVnrArea'].fillna(test['MasVnrArea'].mean())

test['GarageCars'] = test['GarageCars'].fillna(test['GarageCars'].mean())

test['GarageArea'] = test['GarageArea'].fillna(test['GarageArea'].mean())

test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mean())

test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mean())
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop('SalePrice', axis=1), train['SalePrice'], test_size=0.3, random_state=101)
# we are going to scale to data



y_train= y_train.values.reshape(-1,1)

y_test= y_test.values.reshape(-1,1)



from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

sc_y = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.fit_transform(X_test)

y_train = sc_X.fit_transform(y_train)

y_test = sc_y.fit_transform(y_test)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train,y_train)

print(lm)
# print the intercept

print(lm.intercept_)
print(lm.coef_)
predictions = lm.predict(X_test)

predictions= predictions.reshape(-1,1)
plt.figure(figsize=(15,8))

plt.scatter(y_test,predictions)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

#plt.show()

r2_score(y_test, predictions)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
from sklearn import ensemble

from sklearn.utils import shuffle

from sklearn.metrics import mean_squared_error, r2_score
params = {'n_estimators': 100, 'max_depth': 4, 'min_samples_split': 2,

          'learning_rate': 0.05, 'loss': 'ls'}

clf = ensemble.GradientBoostingRegressor(**params)



clf.fit(X_train, y_train)
clf_pred=clf.predict(X_test)

clf_pred= clf_pred.reshape(-1,1)
print('MAE:', metrics.mean_absolute_error(y_test, clf_pred))

print('MSE:', metrics.mean_squared_error(y_test, clf_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, clf_pred)))
plt.figure(figsize=(15,8))

plt.scatter(y_test,clf_pred, c= 'brown')

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

r2_score(y_test, clf_pred)

#plt.show()
from sklearn.tree import DecisionTreeRegressor

dtreg = DecisionTreeRegressor(random_state = 100)

dtreg.fit(X_train, y_train)

dtr_pred = dtreg.predict(X_test)

dtr_pred= dtr_pred.reshape(-1,1)
print('MAE:', metrics.mean_absolute_error(y_test, dtr_pred))

print('MSE:', metrics.mean_squared_error(y_test, dtr_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, dtr_pred)))
plt.figure(figsize=(15,8))

plt.scatter(y_test,dtr_pred,c='green')

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

#plt.show()

r2_score(y_test, dtr_pred)
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators = 100, random_state = 0)

rfr.fit(X_train, y_train)

rfr_pred= rfr.predict(X_test)

rfr_pred = rfr_pred.reshape(-1,1)
print('MAE:', metrics.mean_absolute_error(y_test, rfr_pred))

print('MSE:', metrics.mean_squared_error(y_test, rfr_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rfr_pred)))
plt.figure(figsize=(15,8))

plt.scatter(y_test,rfr_pred, c='orange')

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

#plt.show()

r2_score(y_test, rfr_pred)
error_rate=np.array([metrics.mean_squared_error(y_test, predictions),metrics.mean_squared_error(y_test, clf_pred),metrics.mean_squared_error(y_test, dtr_pred),metrics.mean_squared_error(y_test, rfr_pred)])
plt.figure(figsize=(16,5))

plt.plot(error_rate)
a = pd.read_csv('../input/test.csv')
test_id = a['Id']

a = pd.DataFrame(test_id, columns=['Id'])
test = sc_X.fit_transform(test)
test.shape
test_prediction_grad_boost=clf.predict(test)

test_prediction_grad_boost= test_prediction_grad_boost.reshape(-1,1)
test_prediction_grad_boost
test_prediction_grad_boost =sc_y.inverse_transform(test_prediction_grad_boost)
test_prediction_grad_boost = pd.DataFrame(test_prediction_grad_boost, columns=['SalePrice'])
result = pd.concat([a,test_prediction_grad_boost], axis=1)
result.to_csv('submission.csv',index=False)
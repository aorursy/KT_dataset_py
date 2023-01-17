import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_log_error

from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.preprocessing import StandardScaler

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



plt.rcParams["figure.figsize"] = (20,20)
test_data = pd.read_csv(r'C:\Users\Сергей\Desktop\ДУШИМ ЗМЕЮ\DATA SCIENCE BITCH!\mail.ru\data\kaggle\House Prices\test (1).csv')

train_data = pd.read_csv(r'C:\Users\Сергей\Desktop\ДУШИМ ЗМЕЮ\DATA SCIENCE BITCH!\mail.ru\data\kaggle\House Prices\train (1).csv')
train_data.drop(['Alley', 'FireplaceQu', 'PoolQC'], inplace=True, axis=1)
test_data.drop(['Alley', 'FireplaceQu', 'PoolQC'], inplace=True, axis=1)
train_data.head()
test_data.head()
train_data.info()
test_data.info()
categ_feat = [i for i in train_data.columns if train_data[i].dtype == 'O']

num_feat = [i for i in train_data.columns if train_data[i].dtype == 'int64' or train_data[i].dtype == 'float64']
sns.heatmap(train_data.corr());
train_data['SalePrice'].hist();
for i, col in enumerate(train_data.columns[:20]):

    plt.subplot(5, 4, i+1)

    plt.scatter(train_data[col], train_data['SalePrice'])

    plt.title(col)
for i, col in enumerate(train_data.columns[20:40]):

    plt.subplot(5, 4, i+1)

    plt.scatter(train_data[col], train_data['SalePrice'])

    plt.title(col)
for i, col in enumerate(train_data.columns[40:60]):

    plt.subplot(5, 4, i+1)

    plt.scatter(train_data[col], train_data['SalePrice'])

    plt.title(col)
for i, col in enumerate(train_data.columns[60:]):

    plt.subplot(5, 4, i+1)

    plt.scatter(train_data[col], train_data['SalePrice'])

    plt.title(col)
train_data['Fence'] = train_data['Fence'].apply(lambda x: 0 if x is np.nan else 1)

test_data['Fence'] = train_data['Fence'].apply(lambda x: 0 if x is np.nan else 1)
train_data['MiscFeature'] = train_data['MiscFeature'].apply(lambda x: 0 if x is np.nan else 1)

test_data['MiscFeature'] = train_data['MiscFeature'].apply(lambda x: 0 if x is np.nan else 1)
for col in categ_feat:

    train_data[col].fillna(train_data[col].mode().get(0), inplace=True)
for col in num_feat:

    train_data[col] = train_data[col].fillna(train_data[col].mean())
for col in categ_feat:

    test_data[col].fillna(test_data[col].mode().get(0), inplace=True)
for col in num_feat[:-1]:

    test_data[col] = test_data[col].fillna(test_data[col].mean())
train_data_dummy = pd.get_dummies(train_data[categ_feat])
test_data_dummy = pd.get_dummies(test_data[categ_feat])
set(train_data_dummy.columns) - set(test_data_dummy.columns)
test_data_dummy.shape
train_data_dummy.shape
train_data_dummy_no_intersection = train_data_dummy.drop(list(set(train_data_dummy.columns) - set(test_data_dummy.columns)), axis=1)
train_data_dummy_no_intersection.shape
X = pd.concat([train_data.drop(categ_feat, axis=1), train_data_dummy_no_intersection], axis=1).drop(['SalePrice', 'Id'], axis=1)
y = train_data['SalePrice']
X_test = pd.concat([test_data.drop(categ_feat, axis=1), test_data_dummy], axis=1).drop('Id', axis=1)
X_test.shape
rf = RandomForestRegressor(random_state=42).fit(X, y)
cross_val_score(rf, X, y, scoring='neg_mean_squared_log_error', cv=5, n_jobs=-1, verbose=True)
submit = pd.DataFrame(rf.predict(X_test), index=test_data['Id'], columns=['SalePrice'])
submit
submit.to_csv('submit_rf.csv')
params = {'n_estimators': range(10, 1010, 10)}
best_rf = GridSearchCV(rf, params, n_jobs=-1, cv=5, verbose=True, scoring='neg_mean_squared_log_error')

best_rf.fit(X, y)
best_rf.best_params_, best_rf.best_score_
submit = pd.DataFrame(best_rf.best_estimator_.predict(X_test), index=test_data['Id'], columns=['SalePrice']).to_csv('submit_rf_tuned.csv')
pd.DataFrame(best_rf.best_estimator_.feature_importances_, index=X.columns).sort_values(by=0, ascending=False)
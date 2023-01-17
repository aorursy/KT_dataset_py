import pandas as pd

import numpy as np



from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LinearRegression, Lasso, Ridge

from sklearn.svm import SVR

from sklearn import metrics

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor



import matplotlib.pyplot as plt

import seaborn as sns
!ls ../input/insurance
insurance = pd.read_csv('../input/insurance/insurance.csv')

insurance.head()
insurance.info()
insurance['sex'] = insurance['sex'].map({'male': 0, 'female': 1})

insurance['smoker'] = insurance['smoker'].map({'yes': 1, 'no': 0})

insurance.head()
insurance.info()
insurance.isnull().sum()
sns.heatmap(insurance.corr(), annot=True)
insurance['region'].unique()
region = pd.get_dummies(insurance['region'])

region.head()
insurance.drop(['region'], axis=1, inplace=True)

insurance = pd.merge(insurance, region, on=insurance.index)

insurance.drop(['key_0'], axis=1, inplace=True)

insurance.head()
_, ax = plt.subplots(figsize=(10,8))

sns.heatmap(insurance.corr(), annot=True, ax=ax)
insurance.drop(['northeast', 'northwest', 'southeast', 'southwest'], axis=1, inplace=True)
insurance['charges'].describe()
sns.distplot(insurance['charges'])
#Feature Selection

#3 features vs all



x = insurance[['age', 'bmi', 'smoker']]



y = insurance['charges']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
lr_model = LinearRegression()

lr_model.fit(x_train, y_train)

y_pred = lr_model.predict(x_test)

print('r2 score: ' + str(metrics.r2_score(y_test, y_pred)))

print('mse: ' + str(metrics.mean_squared_error(y_test, y_pred)))
x = insurance.drop(['charges'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
lr_model = LinearRegression()

lr_model.fit(x_train, y_train)

y_pred = lr_model.predict(x_test)

print('r2 score: ' + str(metrics.r2_score(y_test, y_pred)))

print('mse: ' + str(metrics.mean_squared_error(y_test, y_pred)))
sns.distplot(insurance['charges'])
#gaussian curve

transformed_charges = np.log(insurance['charges'])

sns.distplot(transformed_charges)
transformed_charges.head()
scaler = StandardScaler()

scaler.fit(insurance)

insurance_normed = pd.DataFrame(scaler.transform(insurance), columns=insurance.columns)

insurance_normed.head()
sns.distplot(insurance)
sns.distplot(insurance_normed)
x = insurance_normed.drop(['charges'], axis=1)

y = insurance_normed['charges']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
lr_model = LinearRegression()

lr_model.fit(x_train, y_train)

y_pred = lr_model.predict(x_test)

print('r2 score: ' + str(metrics.r2_score(y_test, y_pred)))

print('mse: ' + str(metrics.mean_squared_error(y_test, y_pred)))
s_model = SVR(kernel='linear')

s_model.fit(x_train, y_train)

y_pred = s_model.predict(x_test)

print('r2 score: ' + str(metrics.r2_score(y_test, y_pred)))

print('mse: ' + str(metrics.mean_squared_error(y_test, y_pred)))
rd_model = Ridge()

rd_model.fit(x_train, y_train)

y_pred = rd_model.predict(x_test)

print('r2 score: ' + str(metrics.r2_score(y_test, y_pred)))

print('mse: ' + str(metrics.mean_squared_error(y_test, y_pred)))
ls_model = Lasso()

ls_model.fit(x_train, y_train)

y_pred = ls_model.predict(x_test)

print('r2 score: ' + str(metrics.r2_score(y_test, y_pred)))

print('mse: ' + str(metrics.mean_squared_error(y_test, y_pred)))
#GridSearch

#Linear Regression



parameters = {

    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],

    'C': [0.001, 0.01, 0.1, 1, 10, 100],

    'tol': [0.001, 0.01, 0.1, 1]

}



s_model = SVR()

s_regressor = GridSearchCV(s_model, param_grid=parameters, scoring='neg_mean_squared_error')

grid_result = s_regressor.fit(x_train, y_train)

print(grid_result.best_params_)
s_model = SVR(C=10, gamma='scale', kernel='rbf', tol=0.001)

s_model.fit(x_train, y_train)

y_pred = s_model.predict(x_test)

print('r2 score: ' + str(metrics.r2_score(y_test, y_pred)))

print('mse: ' + str(metrics.mean_squared_error(y_test, y_pred)))
gbr_model = GradientBoostingRegressor(n_estimators=3, max_depth=3, learning_rate=1, criterion='mse', random_state=1)

gbr_model.fit(x_train, y_train)

y_pred = gbr_model.predict(x_test)

print('r2 score: ' + str(metrics.r2_score(y_test, y_pred)))

print('mse: ' + str(metrics.mean_squared_error(y_test, y_pred)))
estimators = [('ridge', Ridge()),

              ('lasso', Lasso()),

              ('svr', SVR()),

              ('lr', LinearRegression())]

reg = StackingRegressor(

    estimators=estimators,

    final_estimator=GradientBoostingRegressor())

reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)

print('r2 score: ' + str(metrics.r2_score(y_test, y_pred)))

print('mse: ' + str(metrics.mean_squared_error(y_test, y_pred)))
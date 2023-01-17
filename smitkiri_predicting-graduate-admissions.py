import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('whitegrid')
data = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')

data.drop('Serial No.', axis=1, inplace=True)
data.head()
data.describe()
plt.hist(data['GRE Score'], bins=25)

plt.show()
plt.scatter(data['GRE Score'], data['TOEFL Score'])

plt.show()
plt.scatter(data['GRE Score'], data['CGPA'])

plt.show()
sns.pairplot(data)
corr = data.corr()

sns.heatmap(corr, annot=True)
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import r2_score

from sklearn.preprocessing import MinMaxScaler
X = data.drop('Chance of Admit ', axis=1).copy()

y = data['Chance of Admit '].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

r2_score(y_test, lr_pred)
rfr = RandomForestRegressor(n_estimators=100)
rfr.fit(X_train, y_train)
rfr_pred = rfr.predict(X_test)

r2_score(y_test, rfr_pred)
imp = pd.DataFrame(sorted(zip(rfr.feature_importances_, X_train.columns), reverse=True), columns=['Importance', 'Feature'])

plt.figure(figsize=(12,6))

sns.barplot(imp['Feature'], imp['Importance'])
scalerX = MinMaxScaler(feature_range=(0, 1))

X_train[X_train.columns] = scalerX.fit_transform(X_train[X_train.columns])

X_test[X_test.columns] = scalerX.transform(X_test[X_test.columns])
rfr = RandomForestRegressor(n_jobs = -1)

param_grid = {'n_estimators': [200, 500, 800, 1000], 

                    'max_depth': [4, 5, 6, 7], 

                    'min_samples_split': [2, 3, 4, 5],

                    'max_features': [1,2,3,4,5,6,7]}

rfr_grid = GridSearchCV(estimator = rfr, param_grid = param_grid, cv = 3,n_jobs = -1)

rfr_grid.fit(X_train,y_train)
rfr_grid.best_params_
rfr.set_params(**rfr_grid.best_params_)
rfr.fit(X_train, y_train)
rfr_pred = rfr.predict(X_test)

r2_score(y_test, rfr_pred)
imp = pd.DataFrame(sorted(zip(rfr.feature_importances_, X_train.columns), reverse=True), columns=['Importance', 'Feature'])

plt.figure(figsize=(12,6))

sns.barplot(imp['Feature'], imp['Importance'])
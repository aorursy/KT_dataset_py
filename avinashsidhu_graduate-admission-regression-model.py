import numpy as np

import pandas as pd
admin_data=pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')

admin_data.drop(['Serial No.'], inplace=True, axis=1)

admin_data.head()
import seaborn as sns

import matplotlib.pyplot as plt

corr = admin_data.corr()

keephalf = np.zeros_like(corr)

keephalf[np.triu_indices_from(keephalf)] = True

plt.figure(figsize=(10,10))

sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f", mask=keephalf);
sns.pairplot(admin_data, corner=True);
y=admin_data['Chance of Admit ']

X=admin_data.drop(['Chance of Admit '], axis=1)
from sklearn.feature_selection import f_regression, mutual_info_regression

mi = mutual_info_regression(X, y)

plt.figure(figsize=(35, 10))

for i in range(0, len(X.columns)):

    plt.subplot(2, len(X.columns), i + 1)

    sns.scatterplot(X.iloc[:, i], y);

    plt.title("MI={:.2f}".format(mi[i]),fontsize=16)

plt.show()
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import Lasso

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split 

from sklearn.metrics import mean_squared_error

from math import sqrt

rmse_rf = sqrt(-1*np.mean(cross_val_score(RandomForestRegressor(n_estimators=10, max_features=2, random_state=0), X, y, cv=10, scoring = 'neg_mean_squared_error')))

rmse_lr = sqrt(-1*np.mean(cross_val_score(LinearRegression(), X, y, cv=10, scoring = 'neg_mean_squared_error')))

rmse_lasso = sqrt(-1*np.mean(cross_val_score(Lasso(alpha=1), X, y, cv=10, scoring = 'neg_mean_squared_error')))
rmse_rf
rmse_lr
rmse_lasso
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

reg=LinearRegression().fit(X_train, y_train)
plt.figure(figsize=(10, 10))

sns.barplot(x=X.columns, y=reg.coef_, palette='colorblind');
rmse = sqrt(mean_squared_error(y.tail(100), reg.predict(X.tail(100))))

rmse
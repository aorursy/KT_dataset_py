# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/beer-consumption-sao-paulo/Consumo_cerveja.csv')
df
df.isnull().sum()
df.dropna(inplace=True)
df.info()
df['Data'] = pd.to_datetime(df['Data'])
df.columns
df['Temperatura Media (C)'] = df['Temperatura Media (C)'].str.replace(',', '.').astype('float')

df['Temperatura Minima (C)'] = df['Temperatura Minima (C)'].str.replace(',', '.').astype('float')

df['Temperatura Maxima (C)'] = df['Temperatura Maxima (C)'].str.replace(',', '.').astype('float')

df['Precipitacao (mm)'] = df['Precipitacao (mm)'].str.replace(',', '.').astype('float')
df.info()
df.describe()
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
sns.pairplot(df)
X = df.drop(columns=['Data', 'Consumo de cerveja (litros)'])

y = df['Consumo de cerveja (litros)']
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()

vif['VIF Factor'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['Features'] = X.columns
vif
X = X.drop(columns='Temperatura Media (C)')

vif = pd.DataFrame()

vif['VIF Factor'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['Features'] = X.columns

vif
X = X.drop(columns='Temperatura Minima (C)')

vif = pd.DataFrame()

vif['VIF Factor'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['Features'] = X.columns

vif
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)

lin_reg_preds = lin_reg.predict(X_test)
print('Linear regression intercept is: ', lin_reg.intercept_)

print('Linear regression coefs are: ', lin_reg.coef_)

print("R squared score for the model is : ", lin_reg.score(X_train, y_train))
sns.scatterplot(y_test, lin_reg_preds)

plt.title('Test data vs Linear Regression Predictions')
sns.distplot(y_test-lin_reg_preds)

plt.title('Residual Distribution')
import statsmodels.api as sm
X_train = sm.add_constant(X_train) # Adding constant



poisson_reg = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
poisson_reg.summary()
X_test = sm.add_constant(X_test)

poisson_reg_preds = poisson_reg.get_prediction(X_test)

poisson_reg_preds.summary_frame()
sns.scatterplot(y_test, poisson_reg_preds.summary_frame()['mean'])

plt.title('Test Data vs Poisson Regression Predictions')
sns.distplot(y_test-poisson_reg_preds.summary_frame()['mean'])

plt.title('Residuals')
nbm_reg = sm.GLM(y_train, X_train, family=sm.families.NegativeBinomial()).fit()

nbm_reg.summary()
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVR
param_grid = {'kernel' : ['linear', 'rbf', 'sigmoid'], 'gamma' : [1, 0.1, 0.01, 0.001], 

              'C' : [1, 10, 100]}
reg_svr = GridSearchCV(estimator=SVR(), param_grid=param_grid)
reg_svr.fit(X_train, y_train)
reg_svr.best_params_
reg_svr = SVR(kernel='linear', C=1, gamma=1)
reg_svr.fit(X_train, y_train)
preds_svr = reg_svr.predict(X_test)
sns.scatterplot(y_test, preds_svr)
reg_svr.score(X_train, y_train)
sns.distplot(y_test - preds_svr)
from sklearn.tree import DecisionTreeRegressor
parameters = {'criterion' : ['mse', 'friedman_mse', 'mae'], 'splitter' : ['random', 'best']}
reg_dtree = GridSearchCV(estimator=DecisionTreeRegressor(), param_grid=parameters)
reg_dtree.fit(X_train, y_train)
reg_dtree.best_params_
reg_dtree = DecisionTreeRegressor(criterion='mae', splitter='best')
reg_dtree.fit(X_train, y_train)
preds_dtree = reg_dtree.predict(X_test)
sns.scatterplot(y_test, preds_dtree)
reg_dtree.score(X_train, y_train)
sns.distplot(y_test - preds_dtree)
from sklearn.ensemble import RandomForestRegressor
p_grid = {'n_estimators' : [100, 200, 500, 800, 1000], 'criterion' : ['mse', 'mae'],

         'min_samples_split' : [3, 4, 5, 6, 7, 8, 9]}
reg_rf = GridSearchCV(estimator=RandomForestRegressor(), param_grid=p_grid)
reg_rf.fit(X_train, y_train)
reg_rf.best_params_
reg_rf = RandomForestRegressor(criterion='mse', min_samples_split=9, n_estimators=800)
reg_rf.fit(X_train, y_train)
pred_rf = reg_rf.predict(X_test)
sns.scatterplot(y_test, pred_rf)
reg_rf.score(X_train, y_train)
sns.distplot(y_test - pred_rf)
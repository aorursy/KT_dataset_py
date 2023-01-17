import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/insurance.csv')

df.head()
df.shape
df.describe()
df.dtypes
df.isnull().sum()
sns.set(style='whitegrid')

f, ax = plt.subplots(1,1, figsize=(12, 8))

ax = sns.distplot(df['charges'], kde = True, color = 'c')

plt.title('Distribution of Charges')
f, ax = plt.subplots(1, 1, figsize=(12, 8))

ax = sns.distplot(np.log10(df['charges']), kde = True, color = 'r' )
charges = df['charges'].groupby(df.region).sum().sort_values(ascending = True)

f, ax = plt.subplots(1, 1, figsize=(8, 6))

ax = sns.barplot(charges.head(), charges.head().index, palette='Blues')
f, ax = plt.subplots(1, 1, figsize=(12, 8))

ax = sns.barplot(x='region', y='charges', hue='sex', data=df, palette='cool')
f, ax = plt.subplots(1,1, figsize=(12,8))

ax = sns.barplot(x = 'region', y = 'charges',

                 hue='smoker', data=df, palette='Reds_r')
f, ax = plt.subplots(1, 1, figsize=(12, 8))

ax = sns.barplot(x='region', y='charges', hue='children', data=df, palette='Set1')
ax = sns.lmplot(x = 'age', y = 'charges', data=df, hue='smoker', palette='Set1')

ax = sns.lmplot(x = 'bmi', y = 'charges', data=df, hue='smoker', palette='Set2')

ax = sns.lmplot(x = 'children', y = 'charges', data=df, hue='smoker', palette='Set3')
f, ax = plt.subplots(1, 1, figsize=(10, 10))

ax = sns.violinplot(x = 'children', y = 'charges', data=df,

                 orient='v', hue='smoker', palette='inferno')
##Converting objects labels into categorical

df[['sex', 'smoker', 'region']] = df[['sex', 'smoker', 'region']].astype('category')

df.dtypes
##Converting category labels into numerical using LabelEncoder

from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

label.fit(df.sex.drop_duplicates())

df.sex = label.transform(df.sex)

label.fit(df.smoker.drop_duplicates())

df.smoker = label.transform(df.smoker)

label.fit(df.region.drop_duplicates())

df.region = label.transform(df.region)

df.dtypes
f, ax = plt.subplots(1, 1, figsize=(10, 10))

ax = sns.heatmap(df.corr(), annot=True, cmap='cool')
from sklearn.model_selection import train_test_split as holdout

from sklearn.linear_model import LinearRegression

from sklearn import metrics

x = df.drop(['charges'], axis = 1)

y = df['charges']

x_train, x_test, y_train, y_test = holdout(x, y, test_size=0.2, random_state=0)

Lin_reg = LinearRegression()

Lin_reg.fit(x_train, y_train)

print(Lin_reg.intercept_)

print(Lin_reg.coef_)

print(Lin_reg.score(x_test, y_test))
from sklearn.linear_model import Ridge

Ridge = Ridge(alpha=0.5)

Ridge.fit(x_train, y_train)

print(Ridge.intercept_)

print(Ridge.coef_)

print(Ridge.score(x_test, y_test))
from sklearn.linear_model import Lasso

Lasso = Lasso(alpha=0.2, fit_intercept=True, normalize=False, precompute=False, max_iter=1000,

              tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')

Lasso.fit(x_train, y_train)

print(Lasso.intercept_)

print(Lasso.coef_)

print(Lasso.score(x_test, y_test))
from sklearn.ensemble import RandomForestRegressor as rfr

x = df.drop(['charges'], axis=1)

y = df.charges

Rfr = rfr(n_estimators = 100, criterion = 'mse',

                              random_state = 1,

                              n_jobs = -1)

Rfr.fit(x_train,y_train)

x_train_pred = Rfr.predict(x_train)

x_test_pred = Rfr.predict(x_test)



print('MSE train data: %.3f, MSE test data: %.3f' % 

      (metrics.mean_squared_error(x_train_pred, y_train),

       metrics.mean_squared_error(x_test_pred, y_test)))

print('R2 train data: %.3f, R2 test data: %.3f' % 

      (metrics.r2_score(y_train,x_train_pred, y_train),

       metrics.r2_score(y_test,x_test_pred, y_test)))
plt.figure(figsize=(8,6))



plt.scatter(x_train_pred, x_train_pred - y_train,

          c = 'gray', marker = 'o', s = 35, alpha = 0.5,

          label = 'Train data')

plt.scatter(x_test_pred, x_test_pred - y_test,

          c = 'blue', marker = 'o', s = 35, alpha = 0.7,

          label = 'Test data')

plt.xlabel('Predicted values')

plt.ylabel('Actual values')

plt.legend(loc = 'upper right')

plt.hlines(y = 0, xmin = 0, xmax = 60000, lw = 2, color = 'red')
print('Feature importance ranking\n\n')

importances = Rfr.feature_importances_

std = np.std([tree.feature_importances_ for tree in Rfr.estimators_],axis=0)

indices = np.argsort(importances)[::-1]

variables = ['age', 'sex', 'bmi', 'children','smoker', 'region']

importance_list = []

for f in range(x.shape[1]):

    variable = variables[indices[f]]

    importance_list.append(variable)

    print("%d.%s(%f)" % (f + 1, variable, importances[indices[f]]))



# Plot the feature importances of the forest

plt.figure()

plt.title("Feature importances")

plt.bar(importance_list, importances[indices],

       color="y", yerr=std[indices], align="center")
from sklearn.preprocessing import PolynomialFeatures

x = df.drop(['charges', 'sex', 'region'], axis = 1)

y = df.charges

pol = PolynomialFeatures (degree = 2)

x_pol = pol.fit_transform(x)

x_train, x_test, y_train, y_test = holdout(x_pol, y, test_size=0.2, random_state=0)

Pol_reg = LinearRegression()

Pol_reg.fit(x_train, y_train)

y_train_pred = Pol_reg.predict(x_train)

y_test_pred = Pol_reg.predict(x_test)

print(Pol_reg.intercept_)

print(Pol_reg.coef_)

print(Pol_reg.score(x_test, y_test))
##Evaluating the performance of the algorithm

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_test_pred))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_test_pred))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
##Predicting the charges

y_test_pred = Pol_reg.predict(x_test)

##Comparing the actual output values with the predicted values

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})

df
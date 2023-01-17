import numpy as np # linear algebra

import pandas as pd # data processing

from math import sqrt

import matplotlib.pyplot as plt 

import seaborn as sns

from scipy.stats import percentileofscore

from scipy import stats

import statsmodels.formula.api as smf

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from scipy.stats import norm

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.metrics import r2_score

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.metrics import mean_squared_error

import os

print(os.listdir("../input"))
df = pd.read_csv('../input/world-happiness-report-2019.csv')
df.head()
df.info()
df.isnull().sum()
df['Positive affect'].fillna(df['Positive affect'].mean(),inplace=True)

df['Negative affect'].fillna(df['Negative affect'].mean(),inplace=True)

df['Social support'].fillna(df['Social support'].mode(),inplace=True)

df['Freedom'].fillna(df['Freedom'].median(),inplace=True)

df['Corruption'].fillna(df['Corruption'].mean(),inplace=True)

df['Generosity'].fillna(df['Generosity'].median(),inplace=True)

df['Log of GDP\nper capita'].fillna(df['Log of GDP\nper capita'].mean(),inplace=True)

df['Healthy life\nexpectancy'].fillna(df['Healthy life\nexpectancy'].mean(),inplace=True)
df.describe()
country_wise = df[['Country (region)', 'Healthy life\nexpectancy']]

country_wise.plot(kind = 'line',figsize=(20,8),color='g')

plt.title('Trend of Healthy Life Expectancy Across the World')

plt.show()
df.corr(method='pearson')
def corrfunc(x, y, **kws):

    r, _ = stats.pearsonr(x, y)

    ax = plt.gca()

    ax.annotate("r = {:.2f}".format(r),

                xy=(.1, .6), xycoords=ax.transAxes,

               size = 24)

    

cmap = sns.cubehelix_palette(light=1, dark = 0.1,

                             hue = 0.5, as_cmap=True)



g = sns.PairGrid(df)



# Scatter plot on the upper triangle

g.map_upper(plt.scatter, s=10, color = 'red')



# Distribution on the diagonal

g.map_diag(sns.distplot, kde=False, color = 'red')



# Density Plot and Correlation coefficients on the lower triangle

g.map_lower(sns.kdeplot, cmap = cmap)

g.map_lower(corrfunc);
features = df.drop(['Corruption', 'Positive affect', 'Negative affect'], axis=1)



corr=df.corr()

plt.figure(figsize=(10, 10))

sns.heatmap(corr, vmax=.8, linewidths=0.01,

            square=True,annot=True,cmap='YlGnBu',linecolor="white")

plt.title('Correlation between features');
features = df.rename(columns={'Healthy life\nexpectancy': 'Healthy Life Exp', 'Log of GDP\nper capita': 'GDP Per Cap'})



features = features.drop('Country (region)', axis=1)

features = features.drop('Generosity', axis=1)

features.head()
sns.distplot(features['Healthy Life Exp'], fit=norm);

fig = plt.figure()

res = stats.probplot(features['Healthy Life Exp'], plot=plt)
y = df['Healthy life\nexpectancy']

X = features.drop('Healthy Life Exp', axis=1)
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state=42)



from sklearn.linear_model import LinearRegression

regressor_normed = LinearRegression(normalize=True)

regressor_normed.fit(X_train, y_train)



y_pred = regressor_normed.predict(X_test)



from sklearn import metrics

print("MAE", metrics.mean_absolute_error(y_test, y_pred))

print("MSE", metrics.mean_squared_error(y_test, y_pred))

print("RMSE", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('r2 score:' , r2_score(y_test,y_pred))
plt.scatter(y_test,y_pred,color='c')

plt.xlabel('y in test')

plt.ylabel('prediction')

plt.title('LinearRegression')
# model reduction using Ridge Regression



rr = Ridge(alpha=0.01)

rr.fit(X_train, y_train)

pred_train_ridge = rr.predict(X_train)



print(np.sqrt(mean_squared_error(y_train, pred_train_ridge)))

print(r2_score(y_train, pred_train_ridge))
pred_test_ridge = rr.predict(X_test)

print("MAE", metrics.mean_absolute_error(y_test, pred_test_ridge))

print("MSE", metrics.mean_squared_error(y_test, pred_test_ridge))

print("RMSE", np.sqrt(metrics.mean_squared_error(y_test, pred_test_ridge)))

print('r2 score:' , r2_score(y_test, pred_test_ridge))
from sklearn.ensemble import RandomForestRegressor 



rf = RandomForestRegressor(n_estimators=1000, random_state=42)



rf.fit(X_train, y_train)



pred_test_rf = rf.predict(X_test)



print("MAE", metrics.mean_absolute_error(y_test, pred_test_rf))

print("MSE", metrics.mean_squared_error(y_test, pred_test_rf))

print("RMSE", np.sqrt(metrics.mean_squared_error(y_test, pred_test_rf)))

print('r2 score:' , r2_score(y_test,pred_test_rf))
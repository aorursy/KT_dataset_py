#importing library

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



warnings.simplefilter("ignore")

%matplotlib inline
data = pd.read_csv("../input/bmi-and-life/bmi_and_life_expectancy.csv")
data.head(2)
data.shape
data.isnull().any()
data.describe()
plt.figure(figsize=(7,7))

sns.boxplot(x= 'variable', y = 'value', data = pd.melt(data[['Life expectancy', 'BMI']]))
# Plotting the heatmap of correlation between features

plt.figure(figsize=(7,7))

sns.heatmap(data.corr(), cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':10}, cmap='Greys')
data.plot.hist(grid=True, bins=20, rwidth=0.9)
sns.distplot(data['BMI'])

plt.title("Histogram of BMI")

plt.xlabel("BMI")

plt.ylabel("Frequency")

plt.show()
sns.distplot(data['Life expectancy'])

plt.title("Histogram for Life Expectancy ")

plt.xlabel("Life expectancy")

plt.ylabel("Frequency")

plt.show()
#spliting the data

from sklearn.model_selection import train_test_split
X = data['Life expectancy']

y = data['BMI']
X = X.values.reshape(-1,1)

y = y.values.reshape(-1,1)
#scaling the data

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_scaled = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size = 0.3, random_state = 100)
from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
lr = LinearRegression()

lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
dtr = DecisionTreeRegressor()

dtr.fit(X_train, y_train)
y_pred_dtr = dtr.predict(X_test)
rfr = RandomForestRegressor()

rfr.fit(X_train, y_train)
y_pred_rfr = rfr.predict(X_test)
xgr = XGBRegressor()

xgr.fit(X_train, y_train)
y_pred_xgr = xgr.predict(X_test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled,y, test_size = 0.3, random_state = 100)
lr = LinearRegression()

lr.fit(X_train, y_train)
y_pred_lrs = lr.predict(X_test)
rfr = RandomForestRegressor()

rfr.fit(X_train, y_train)
y_pred_rfrs = rfr.predict(X_test)
dtr = DecisionTreeRegressor()

dtr.fit(X_train, y_train)
y_pred_dtrs = dtr.predict(X_test)
xgr = XGBRegressor()

xgr.fit(X_train, y_train)
y_pred_xgrs = xgr.predict(X_test)
from sklearn.metrics import r2_score
R2_score_LR = r2_score(y_test, y_pred_lr)
R2_score_DTR =  r2_score(y_test, y_pred_dtr)
R2_score_RFR = r2_score(y_test, y_pred_rfr)
R2_score_xgr = r2_score(y_test, y_pred_xgr)
R2_score_LRs = r2_score(y_test, y_pred_lrs)
R2_score_DTRs = r2_score(y_test, y_pred_dtrs)
R2_score_RFRs = r2_score(y_test, y_pred_rfrs)
R2_score_xgrs = r2_score(y_test, y_pred_xgrs)
models = pd.DataFrame({

    'Model': ['Linear Regression', 'Random Forest', 'XGBoost', 'Decision Tree'],

    'R-square(Non-scaled)': [R2_score_LR*100, R2_score_RFR*100, R2_score_xgr*100, R2_score_DTR*100],

    'R-square(Scaled)': [R2_score_LRs*100, R2_score_RFRs*100, R2_score_xgrs*100, R2_score_DTRs*100]})

models.sort_values(by = 'Model',ascending=False)
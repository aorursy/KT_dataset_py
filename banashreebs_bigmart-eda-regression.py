# BigMart Sales Prediction

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

from scipy.stats import norm

from scipy import stats



# Load the data 

a = pd.read_csv(r'../input/big-mart-sales-prediction/Train.csv')

b = pd.read_csv(r'../input/big-mart-sales-prediction/Test.csv')

# I store the ID for later use and delete it from the data.

c = b.iloc[:, 0]

d = b.iloc[:, 6]



a.drop(['Item_Identifier', 'Outlet_Identifier'], axis = 1, inplace = True)

b.drop(['Item_Identifier', 'Outlet_Identifier'], axis = 1, inplace = True)
categorical = ['Item_Fat_Content', 'Outlet_Size','Outlet_Location_Type','Outlet_Type', 'Item_Type']

continuous = ['Item_Weight','Item_Visibility', 'Item_MRP','Item_Outlet_Sales']
a.info()

b.info()
sns.distplot(a['Item_Outlet_Sales'])

plt.show()
a.head()
a.isnull().sum()
b.isnull().sum()
a.eq(0).sum()
b.eq(0).sum()
from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values  = np.nan, strategy = 'mean')

a.iloc[:, [0]] = imp.fit_transform(a.iloc[:,[0]])

b.iloc[:, [0]] = imp.transform(b.iloc[:,[0]])

imp1 = SimpleImputer(missing_values  = 0, strategy = 'mean')

a.iloc[:, [2]] = imp1.fit_transform(a.iloc[:,[2]])

b.iloc[:, [2]] = imp1.transform(b.iloc[:,[2]])
a['Outlet_Size'].fillna(a['Outlet_Size'].mode()[0], inplace = True)

b['Outlet_Size'].fillna(b['Outlet_Size'].mode()[0], inplace = True)
sns.heatmap(a.corr(), annot = True)

plt.show()
sns.scatterplot(x = a['Item_MRP'], y = a['Item_Outlet_Sales'], data = a)

plt.show()
sns.pairplot(a)

plt.show()
a['Outlet_Establishment_Year'].unique()
fig, axes = plt.subplots(nrows=1, ncols=3, figsize = (15,8))

fig.subplots_adjust(right=1)

fig.suptitle('Relationshp with Categorical Features')

for ax, feature in zip(axes.flatten(),  categorical[0:3]):

    sns.stripplot(x = feature,  y = 'Item_Outlet_Sales', data = a, ax = ax)

plt.show()
a.Item_Fat_Content = a.Item_Fat_Content.replace({'low fat' : 'Low Fat', 'LF' : 'Low Fat', 'reg' : 'Regular'})

b.Item_Fat_Content = b.Item_Fat_Content.replace({'low fat' : 'Low Fat', 'LF' : 'Low Fat', 'reg' : 'Regular'})

fig, axes = plt.subplots(nrows=1, ncols=3, figsize = (15,8))

fig.subplots_adjust(right=1)

fig.suptitle('Relationshp with Categorical Data')

for ax, feature in zip(axes.flatten(),  categorical[0:3]):

    sns.stripplot(x = feature,  y = 'Item_Outlet_Sales', data = a, ax = ax)

plt.show()
fig, axes = plt.subplots(nrows = 2, ncols=1, figsize = (20,25))

fig.subplots_adjust(hspace=0.5)

fig.suptitle('Relationshp with Categorical Data')

for ax, feature in zip(axes.flatten(),  categorical[3:]):

    sns.stripplot(x = 'Item_Outlet_Sales', y = feature, data = a, ax = ax)

plt.show()
fig,axes = plt.subplots(figsize = (10,10))

sns.boxplot(x = a['Outlet_Establishment_Year'], y = a['Item_Outlet_Sales'], hue = a['Outlet_Type'], ax = axes )

plt.plot
# Encoding Categorical

a = pd.get_dummies(a, drop_first = True)

b = pd.get_dummies(b, drop_first = True)
a.info()
# Splitting

Y = a['Item_Outlet_Sales']

X = a.drop('Item_Outlet_Sales', axis = 1)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 0, test_size = 0.25)

# Model

# Linear Regression

from sklearn.linear_model import LinearRegression

lg = LinearRegression()

lg.fit(X_train, Y_train)

Y_pred = lg.predict(X_test)

residue = Y_test - Y_pred

sns.regplot(residue, Y_pred, lowess = True, line_kws={'color': 'red'})

plt.show()

Y_train = np.log(Y_train)

Y_test = np.log(Y_test)
from sklearn.linear_model import LinearRegression

lg = LinearRegression()

lg.fit(X_train, Y_train)

Y_pred = lg.predict(X_test)

residue = Y_test - Y_pred

sns.regplot(residue, Y_pred, lowess = True, line_kws={'color': 'red'})

plt.show()

# SVM

from sklearn.svm import SVR

regressor = SVR(kernel = 'rbf')

regressor.fit(X_train, Y_train)

Y_pred2 = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error

from math import sqrt

rms = sqrt(mean_squared_error(Y_test, Y_pred2))

from sklearn.metrics import r2_score

r2 = r2_score(Y_test, Y_pred2)

print('RMSE = ',rms, ' R2 score = ',r2)
from sklearn.linear_model import LassoCV

model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005])

model_lasso.fit(X_train, Y_train)

coef = pd.Series(model_lasso.coef_, index = X_train.columns)

imp_features = coef.index[coef!=0].tolist()



imp_features

X_train = X_train[imp_features]

X_test = X_test[imp_features]



from sklearn.preprocessing import KBinsDiscretizer

disc = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

X_train['Outlet_Establishment_Year'] = disc.fit_transform(X_train[['Outlet_Establishment_Year']])

X_test['Outlet_Establishment_Year'] = disc.fit_transform(X_test[['Outlet_Establishment_Year']])

from sklearn.svm import SVR

regressor = SVR(kernel = 'rbf')

regressor.fit(X_train, Y_train)

Y_pred3 = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error

from math import sqrt

rms = sqrt(mean_squared_error(Y_test, Y_pred3))

from sklearn.metrics import r2_score

r2 = r2_score(Y_test, Y_pred3)

print('RMSE = ',rms, ' R2 score = ',r2)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train.iloc[:,0:2] = sc_X.fit_transform(X_train.iloc[:,0:2])

X_test.iloc[:,0:2] = sc_X.transform(X_test.iloc[:,0:2])

from sklearn.svm import SVR

regressor = SVR(kernel = 'rbf')

regressor.fit(X_train, Y_train)

Y_pred3 = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error

from math import sqrt

rms = sqrt(mean_squared_error(Y_test, Y_pred3))

from sklearn.metrics import r2_score

r2 = r2_score(Y_test, Y_pred3)

print('RMSE = ',rms, ' R2 score = ',r2)
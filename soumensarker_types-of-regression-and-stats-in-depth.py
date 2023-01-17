import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1.5, color_codes=True)

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

#plt.style.use('ggplot')

#ggplot is R based visualisation package that provides better graphics with higher level of abstraction

import os
diamond_data = pd.read_csv("../input/diamonds.csv")
diamond_data.info()
diamond_data.head()
diamond_data = diamond_data.drop(["Unnamed: 0"],axis=1)

diamond_data.head()
plt.figure(figsize=(20,20))  # on this line I just set the size of figure to 12 by 10.

p=sns.heatmap(diamond_data.corr(), annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap
p=sns.pairplot(diamond_data)
diamond_data.describe()
print("Number of rows with x == 0: {} ".format((diamond_data.x==0).sum()))

print("Number of rows with y == 0: {} ".format((diamond_data.y==0).sum()))

print("Number of rows with z == 0: {} ".format((diamond_data.z==0).sum()))

print("Number of rows with depth == 0: {} ".format((diamond_data.depth==0).sum()))
diamond_data[['x','y','z']] = diamond_data[['x','y','z']].replace(0,np.NaN)
diamond_data.isnull().sum()
diamond_data.dropna(inplace=True)
diamond_data.shape
diamond_data.isnull().sum()
p = diamond_data.hist(figsize = (20,20),bins=150)
p = sns.factorplot(x='cut', data=diamond_data , kind='count',aspect=2.5 )
p = sns.factorplot(x='cut', y='price', data=diamond_data, kind='box' ,aspect=2.5 )
p = diamond_data.hist(figsize = (20,20), by=diamond_data.cut,grid=True)
p = sns.factorplot(x='color', data=diamond_data , kind='count',aspect=2.5 )
p = sns.factorplot(x='color', y='price', data=diamond_data, kind='box' ,aspect=2.5 )
p = sns.factorplot(x='clarity', data=diamond_data , kind='count',aspect=2.5 )
p = sns.factorplot(x='clarity', y='price', data=diamond_data, kind='box' ,aspect=2.5)
one_hot_encoders_diamond_data =  pd.get_dummies(diamond_data)

one_hot_encoders_diamond_data.head()
# a structured approach

cols = one_hot_encoders_diamond_data.columns

diamond_clean_data = pd.DataFrame(one_hot_encoders_diamond_data,columns= cols)

diamond_clean_data.head()
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

numericals =  pd.DataFrame(sc_X.fit_transform(diamond_clean_data[['carat','depth','x','y','z','table']]),columns=['carat','depth','x','y','z','table'],index=diamond_clean_data.index)
numericals.head()
diamond_clean_data_standard = diamond_clean_data.copy(deep=True)

diamond_clean_data_standard[['carat','depth','x','y','z','table']] = numericals[['carat','depth','x','y','z','table']]
diamond_clean_data_standard.head()
plt.figure(figsize=(20,20))  # on this line I just set the size of figure to 12 by 10.

p=sns.heatmap(diamond_clean_data.corr(), annot=True,cmap='RdYlGn')  # seaborn has very simple solution for heatmap
plt.figure(figsize=(20,20))  # on this line I just set the size of figure to 12 by 10.

p=sns.heatmap(diamond_clean_data_standard.corr(), annot=True,cmap='RdYlGn')  # seaborn has very simple solution for heatmap
x = diamond_clean_data_standard.drop(["price"],axis=1)

y = diamond_clean_data_standard.price
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(x, y,random_state = 2,test_size=0.3)
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn import linear_model



regr = linear_model.LinearRegression()

regr.fit(train_x,train_y)

y_pred = regr.predict(test_x)

print("accuracy: "+ str(regr.score(test_x,test_y)*100) + "%")

print("Mean absolute error: {}".format(mean_absolute_error(test_y,y_pred)))

print("Mean squared error: {}".format(mean_squared_error(test_y,y_pred)))

R2 = r2_score(test_y,y_pred)

print('R Squared: {}'.format(R2))

n=test_x.shape[0]

p=test_x.shape[1] - 1



adj_rsquared = 1 - (1 - R2) * ((n - 1)/(n-p-1))

print('Adjusted R Squared: {}'.format(adj_rsquared))
las_reg = linear_model.Lasso()

las_reg.fit(train_x,train_y)

y_pred = las_reg.predict(test_x)

print("accuracy: "+ str(las_reg.score(test_x,test_y)*100) + "%")

print("Mean absolute error: {}".format(mean_absolute_error(test_y,y_pred)))

print("Mean squared error: {}".format(mean_squared_error(test_y,y_pred)))

R2 = r2_score(test_y,y_pred)

print('R Squared: {}'.format(R2))

n=test_x.shape[0]

p=test_x.shape[1] - 1



adj_rsquared = 1 - (1 - R2) * ((n - 1)/(n-p-1))

print('Adjusted R Squared: {}'.format(adj_rsquared))
rig_reg = linear_model.Ridge()

rig_reg.fit(train_x,train_y)

y_pred = rig_reg.predict(test_x)

print("accuracy: "+ str(rig_reg.score(test_x,test_y)*100) + "%")

print("Mean absolute error: {}".format(mean_absolute_error(test_y,y_pred)))

print("Mean squared error: {}".format(mean_squared_error(test_y,y_pred)))

R2 = r2_score(test_y,y_pred)

print('R Squared: {}'.format(R2))

n=test_x.shape[0]

p=test_x.shape[1] - 1



adj_rsquared = 1 - (1 - R2) * ((n - 1)/(n-p-1))

print('Adjusted R Squared: {}'.format(adj_rsquared))
l = list(range(0,len(diamond_clean_data_standard.columns)))
import statsmodels.formula.api as smf

X = np.append(arr = np.ones((diamond_clean_data_standard.shape[0], 1)).astype(int), values = diamond_clean_data_standard.drop(['price'],axis=1).values, axis = 1)

X_opt = X[:, l]

regressor_ols = smf.OLS(endog = diamond_clean_data_standard.price, exog = X_opt).fit()

regressor_ols.summary()
l.pop(5)

X = np.append(arr = np.ones((diamond_clean_data_standard.shape[0], 1)).astype(int), values = diamond_clean_data_standard.drop(['price'],axis=1).values, axis = 1)

X_opt = X[:, l]

regressor_ols = smf.OLS(endog = diamond_clean_data_standard.price, exog = X_opt).fit()

regressor_ols.summary()
l.pop(5)

X = np.append(arr = np.ones((diamond_clean_data_standard.shape[0], 1)).astype(int), values = diamond_clean_data_standard.drop(['price'],axis=1).values, axis = 1)

X_opt = X[:, l]

regressor_ols = smf.OLS(endog = diamond_clean_data_standard.price, exog = X_opt).fit()

regressor_ols.summary()
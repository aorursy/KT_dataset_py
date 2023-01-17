# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
test.head(2)
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
train.head(2)
train.columns
nandetails = train.isnull().sum()
d = dict(zip(nandetails.index,nandetails.values))
a = sorted(d.items(), key=lambda x: x[1],reverse = True)    

print(a)
f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(train.corr(), vmax=.8, square=True,cmap= 'coolwarm')
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
nt = train[cols]
f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(nt.corr(), vmax=.8,annot = True, square=True,cmap= 'coolwarm')
sns.pairplot(nt, height = 4)
nt.head()
nt.describe()
X = nt.drop(['SalePrice'], axis=1)
X.head(2)
y= nt.SalePrice
X.isnull().sum()
X.shape
from sklearn.model_selection import train_test_split



from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_squared_error



from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
import math

lr_model = LinearRegression()

lr_model.fit(X_train, y_train)



print('Training score: {}'.format(lr_model.score(X_train, y_train)))

print('Test score: {}'.format(lr_model.score(X_test, y_test)))



y_pred = lr_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

rmse = math.sqrt(mse)



print('RMSE: {}'.format(rmse))
steps = [

    ('scalar', StandardScaler()),

    ('poly', PolynomialFeatures(degree=2)),

    ('model', LinearRegression())

]



pipeline = Pipeline(steps)



pipeline.fit(X_train, y_train)



print('Training score: {}'.format(pipeline.score(X_train, y_train)))

print('Test score: {}'.format(pipeline.score(X_test, y_test)))
steps = [

    ('scalar', StandardScaler()),

    ('poly', PolynomialFeatures(degree=2)),

    ('model', Ridge(alpha=10, fit_intercept=True))

]



ridge_pipe = Pipeline(steps)

ridge_pipe.fit(X_train, y_train)



print('Training Score: {}'.format(ridge_pipe.score(X_train, y_train)))

print('Test Score: {}'.format(ridge_pipe.score(X_test, y_test)))
from sklearn.pipeline import make_pipeline
t = make_pipeline(StandardScaler(),PolynomialFeatures(degree=2),Ridge(alpha=10, fit_intercept=True)).fit(X_train, y_train)
print('Training Score: {}'.format(t.score(X_train, y_train)))

print('Test Score: {}'.format(t.score(X_test, y_test)))
lasso = make_pipeline(StandardScaler(),PolynomialFeatures(degree=2),Lasso(alpha=0.1)).fit(X_train, y_train)
print('Training Score: {}'.format(lasso.score(X_train, y_train)))

print('Test Score: {}'.format(lasso.score(X_test, y_test)))
en = make_pipeline(StandardScaler(),PolynomialFeatures(degree=2),ElasticNet(alpha=0.09)).fit(X_train, y_train)
print('Training Score: {}'.format(en.score(X_train, y_train)))

print('Test Score: {}'.format(en.score(X_test, y_test)))
y_pred = en.predict(X_test)
from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

print(np.sqrt(mean_squared_error(y_test,y_pred ))) 

print(r2_score(y_test, y_pred ))
fig, ((ax1, ax2), (ax3, ax4),(ax5,ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(14,10))

OverallQual_scatter_plot = pd.concat([train['SalePrice'],train['OverallQual']],axis = 1)

sns.regplot(x='OverallQual',y = 'SalePrice',data = OverallQual_scatter_plot,scatter= True, fit_reg=True, ax=ax1)

TotalBsmtSF_scatter_plot = pd.concat([train['SalePrice'],train['TotalBsmtSF']],axis = 1)

sns.regplot(x='TotalBsmtSF',y = 'SalePrice',data = TotalBsmtSF_scatter_plot,scatter= True, fit_reg=True, ax=ax2)

GrLivArea_scatter_plot = pd.concat([train['SalePrice'],train['GrLivArea']],axis = 1)

sns.regplot(x='GrLivArea',y = 'SalePrice',data = GrLivArea_scatter_plot,scatter= True, fit_reg=True, ax=ax3)

GarageArea_scatter_plot = pd.concat([train['SalePrice'],train['GarageArea']],axis = 1)

sns.regplot(x='GarageArea',y = 'SalePrice',data = GarageArea_scatter_plot,scatter= True, fit_reg=True, ax=ax4)

FullBath_scatter_plot = pd.concat([train['SalePrice'],train['FullBath']],axis = 1)

sns.regplot(x='FullBath',y = 'SalePrice',data = FullBath_scatter_plot,scatter= True, fit_reg=True, ax=ax5)

YearBuilt_scatter_plot = pd.concat([train['SalePrice'],train['YearBuilt']],axis = 1)

sns.regplot(x='YearBuilt',y = 'SalePrice',data = YearBuilt_scatter_plot,scatter= True, fit_reg=True, ax=ax6)

YearRemodAdd_scatter_plot = pd.concat([train['SalePrice'],train['YearRemodAdd']],axis = 1)

YearRemodAdd_scatter_plot.plot.scatter('YearRemodAdd','SalePrice')
saleprice_overall_quality= train.pivot_table(index ='OverallQual',values = 'SalePrice', aggfunc = np.median)

saleprice_overall_quality.plot(kind = 'bar',color = 'blue')

plt.xlabel('Overall Quality')

plt.ylabel('Median Sale Price')

plt.show()
var = 'OverallQual'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(12, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000)
# plot missing values in bar
train_d = train.select_dtypes(include=[np.object])



missing_values = train_d.isnull().sum(axis=0).reset_index()

missing_values.columns = ['column_name', 'missing_count']

missing_values = missing_values.loc[missing_values['missing_count']>0]

missing_values = missing_values.sort_values(by='missing_count')



ind = np.arange(missing_values.shape[0])

width = 0.9

fig, ax = plt.subplots(figsize=(12,18))

rects = ax.barh(ind, missing_values.missing_count.values, color='red')

ax.set_yticks(ind)

ax.set_yticklabels(missing_values.column_name.values, rotation='horizontal')

ax.set_xlabel("Missing Observations Count")

ax.set_title("Missing Observations Count - Categorical Features")

plt.show()
# violinplot of train data of column functional and saleprice

sns.violinplot('Functional', 'SalePrice', data = train)
var = 'Neighborhood'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

plt.figure(figsize = (12, 6))

sns.countplot(x = 'Neighborhood', data = data)

xt = plt.xticks(rotation=45)

var = 'SaleCondition'



data = pd.concat([train['SalePrice'], train[var]], axis=1)



f, ax = plt.subplots(figsize=(16, 10))



fig = sns.boxplot(x=var, y="SalePrice", data=data)



fig.axis(ymin=0, ymax=800000);



xt = plt.xticks(rotation=45)
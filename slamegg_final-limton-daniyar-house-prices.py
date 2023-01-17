# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running tThe most convenient way to take a quick look at a univariate distribution in seaborn is the distplot() function. By default, this will draw a histogram and fit a kernel density estimatehis (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
import seaborn as sns
print(train_df.shape, test_df.shape)
train_df.head(5)
test_df.head(5)
train_df.info()
test_df.info()
sns.distplot(train_df.SalePrice)
import matplotlib.pyplot as plt

plt.figure(figsize=(18,4))

ax = sns.countplot(x="YrSold", data=train_df)

ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

for p in ax.patches:

    ax.annotate(format(p.get_height()), 

               (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center',rotation=30, va = 'center', xytext = (0, 10), textcoords = 'offset points')
import matplotlib.pyplot as plt

plt.figure(figsize=(18,4))

ax = sns.countplot(x="YrSold", data=test_df)

ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

for p in ax.patches:

    ax.annotate(format(p.get_height()), 

               (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center',rotation=0, va = 'center', xytext = (0, 10), textcoords = 'offset points')
d = {'LotArea': [train_df.LotArea.mean(), test_df.LotArea.mean()], 'name': ['Train','Test']}

df = pd.DataFrame(data=d, dtype=np.float64)



plt.figure(figsize=(8,4)) 

splot = sns.barplot(data=df, x = 'name', y = 'LotArea', ci = None) 

for p in splot.patches:

    splot.annotate(format(p.get_height(), '.2f'), 

               (p.get_x() + p.get_width() / 2., p.get_height()), 

                   ha = 'center', va = 'center', xytext = (0, -20), textcoords = 'offset points')
train_df.describe(include = 'all')
train_df.nunique()
train_df.isna().sum()
test_df.isna().sum()
train_df.info()
#descriptive statistics summary

train_df['SalePrice'].describe()
print("Skewness: %f" % train_df['SalePrice'].skew())

print("Kurtosis: %f" % train_df['SalePrice'].kurt())
var = 'GrLivArea'

data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
corr_matrix = train_df.corr()

corr_matrix["SalePrice"].sort_values(ascending=False)
object_list = list(train_df.select_dtypes(include=['object']).columns)

object_list

dummies = pd.get_dummies(train_df[object_list])

dummies
train_df = pd.concat([train_df, dummies], axis=1)

train_df = train_df.drop(object_list, axis=1)
train_df.isna().sum()
with pd.option_context('display.max_rows', None, 'display.max_columns', None): 

  print(train_df.isna().sum())
train_df.LotFrontage.fillna(train_df.LotFrontage.mean(), inplace=True)

train_df.GarageYrBlt.fillna(train_df.GarageYrBlt.mean(), inplace=True)

train_df.MasVnrArea.fillna(train_df.MasVnrArea.mean(), inplace=True)
with pd.option_context('display.max_rows', None, 'display.max_columns', None): 

  print(train_df.isna().sum())
train_df['SalePrice'].describe()
numerical=train_df.dtypes[train_df.dtypes!= 'object'].index

categorical=train_df.dtypes[train_df.dtypes== 'object'].index

corr=train_df[numerical].corr()

corr
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score

from sklearn.impute import SimpleImputer
X_train = train_df.drop('SalePrice', axis=1)

y_train = train_df.SalePrice

X_test = test_df
object_list2 = list(test_df.select_dtypes(include=['object']).columns)

object_list2
dummies2 = pd.get_dummies(test_df[object_list2])

dummies2
test_df = pd.concat([test_df, dummies2], axis=1)

test_df = test_df.drop(object_list2, axis=1)
with pd.option_context('display.max_rows', None, 'display.max_columns', None): 

  print(test_df.isna().sum())
test_df = test_df.fillna(test_df.mean())

train_df.shape
reg = LinearRegression()

X_test = test_df
onehot_train_X = pd.get_dummies(X_train)

onehot_test_X = pd.get_dummies(X_test)
X_train, X_test = onehot_train_X.align(onehot_test_X, join='left', axis=1)

my_imputer = SimpleImputer()

X_train = my_imputer.fit_transform(X_train)

X_test = my_imputer.transform(X_test)
reg = LinearRegression()

reg.fit(X_train, y_train)

y_preds = reg.predict(X_test)
submission_linreg = pd.DataFrame({'Id': test_df.Id, 'SalePrice':y_preds})
submission_linreg.to_csv('submission_linreg.csv', index=False)
submission_linreg.head()
from sklearn.ensemble import GradientBoostingRegressor


params = {'n_estimators': 3000, 'max_depth': 1, 'min_samples_leaf':15, 'min_samples_split':10, 

          'learning_rate': 0.05, 'loss': 'huber','max_features':'sqrt'}

gbr = GradientBoostingRegressor(**params)

gbr.fit(X_train, y_train)
y_grad_predict = gbr.predict(X_test)
dataset_grb = pd.DataFrame({'Id': test_df.Id, 'SalePrice': y_grad_predict})





dataset_grb.to_csv('submission_grb2.csv', encoding='utf-8', index=False)
# def rmse_cv(model):

#     rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 3))

#     return(rmse)
# n_estimators = [3000, 100, 1000]

# cv_rmse_gb = [rmse_cv(GradientBoostingRegressor(n_estimators = n_estimator)).mean() 

#             for n_estimator in n_estimators]
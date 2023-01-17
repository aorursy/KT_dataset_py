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



import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats
train_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train_df.columns
train_df.info()
train_df.describe()
sns.distplot(train_df['SalePrice'])
print("Skewness is: %f" % train_df['SalePrice'].skew())

print("Kurtosis is: %f" % train_df['SalePrice'].kurt())
corrmat = train_df.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True)
k = 11 

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

corr = np.corrcoef(train_df[cols].values.T)

heat = sns.heatmap(corr, yticklabels=cols.values, xticklabels=cols.values)
df_cm = pd.DataFrame(train_df[cols])



def corrfunc(x, y, **kws):

    r, _ = stats.pearsonr(x, y)

    ax = plt.gca()

    ax.annotate("r = {:.2f}".format(r),

                xy=(.1, .6), xycoords=ax.transAxes,

               size = 24)

    

cmap = sns.cubehelix_palette(light=1, dark = 0.1,

                             hue = 0.5, as_cmap=True)



g = sns.PairGrid(df_cm)



# Scatter plot on the upper triangle

g.map_upper(plt.scatter, s=10, color = 'red')



# Distribution on the diagonal

g.map_diag(sns.distplot, kde=False, color = 'red')



# Density Plot and Correlation coefficients on the lower triangle

g.map_lower(sns.kdeplot, cmap = cmap)

g.map_lower(corrfunc);
numeric_features = train_df.select_dtypes(include=[np.number])

numeric_features.dtypes
numeric_features.describe()
corr = numeric_features.corr()

print(corr['SalePrice'].sort_values(ascending=False)[:15], '\n')
overallqual = pd.pivot_table(train_df, index='OverallQual', values='SalePrice', aggfunc=np.median)

overallqual.plot(kind='bar')
plt.scatter(train_df['GrLivArea'], train_df['SalePrice'])

plt.scatter(train_df['GarageArea'], train_df['SalePrice'])

categoricals = train_df.select_dtypes(exclude=[np.number])

categoricals.describe()
train_df = pd.get_dummies(train_df)

train_df = train_df.interpolate().dropna()
y = np.log(train_df.SalePrice)

X = train_df.drop(['SalePrice', 'Id'], axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

                          X, y, random_state=42, test_size=.33)
from sklearn import linear_model

lr = linear_model.LinearRegression()

model = lr.fit(X_train, y_train)

predictions = model.predict(X_train)



from sklearn.metrics import mean_squared_error

print ("R^2 is: \n", model.score(X_train, y_train))

print ('RMSE is: \n', mean_squared_error(y_train, predictions))
#test 



predictions = model.predict(X_test)



from sklearn.metrics import mean_squared_error

print ("R^2 is: \n", model.score(X_test, y_test))

print ('RMSE is: \n', mean_squared_error(y_test, predictions))
for i in range (-2, 3):

    alpha = 10**i

    rm = linear_model.Ridge(alpha=alpha)

    ridge_model = rm.fit(X_train, y_train)

    preds_ridge = ridge_model.predict(X_test)

    print('R^2 is: {}\nRMSE is: {}'.format(

                    ridge_model.score(X_test, y_test),

                    mean_squared_error(y_test, preds_ridge)))
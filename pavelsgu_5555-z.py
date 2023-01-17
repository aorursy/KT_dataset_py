# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
ice = pd.read_csv('../input/SeaIce.txt', delim_whitespace = True)
print ('shape:', ice.shape)
ice.head()

ice2 = ice[ice.data_type != '-9999']
import seaborn as sns
sns.lmplot("mo", "extent", ice2)
month_means=ice2[["extent","mo"]].groupby("mo").mean()
for i in range(12):
   ice2.extent[ice2.mo == i+1] =100*(ice2.extent[ice2.mo == i+1]- month_means.extent[i+1])/month_means.extent.mean()
sns.lmplot("mo", "extent", ice2)
sns.lmplot("year", "extent", ice2)
from sklearn.linear_model import LinearRegression
est = LinearRegression(fit_intercept = True)
x = ice2[['year']]
y = ice2[['extent']]
est.fit(x, y)
print ("Coefficients:", est.coef_)
print ("Intercept:", est.intercept_)
from sklearn import metrics
y_hat = est.predict(x)
print ("MSE:", metrics.mean_squared_error(y_hat , y))
print ("R^2:", metrics.r2_score(y_hat , y))
print ('var:', y.var())
x = [[2025]]
y_hat = est.predict(x)
m = 1 # January
y_hat = (y_hat*month_means.extent.mean()/100) + month_means.extent[m]
print ("Prediction of extent for January 2025 (in millions of square km):", y_hat)
from sklearn import datasets
boston = datasets.load_boston()
X_boston , y_boston = boston.data , boston.target
print ('Shape of data:', X_boston.shape , y_boston.shape)
print ('Feature names:',boston.feature_names)
df_boston = pd.DataFrame(boston.data ,
columns = boston.feature_names)
df_boston['price'] = boston.target
sns.lmplot("price", "LSTAT", df_boston)
sns.lmplot("price", "LSTAT", df_boston , order = 2)
indexes = [0,2,4,5,6,12]
df2 = pd.DataFrame(boston.data[:,indexes],
columns = boston.feature_names[indexes])
df2['price'] = boston.target
corrmat = df2.corr()
sns.heatmap(corrmat , vmax = .8, square = True)
indexes=[5,6,12]
df2 = pd.DataFrame(boston.data[:,indexes],columns = boston.feature_names[indexes])
df2['price'] = boston.target
pd.plotting.scatter_matrix(df2, figsize = (12.0, 12.0))
from sklearn import linear_model
train_size = X_boston.shape [0]/2
train_size=int(train_size)
X_train = X_boston[:train_size]
X_test = X_boston[train_size:]
y_train = y_boston[:train_size]
y_test = y_boston[train_size:]
print ('Training and testing set sizes ',X_train.shape, X_test.shape, X_train.shape , X_test.shape)
regr = LinearRegression()
regr.fit(X_train , y_train)
print ('Coeff and intercept:',regr.coef_ , regr.intercept_)
print ('Testing Score:', regr.score(X_test , y_test))
print ('Training MSE:', np.mean((regr.predict(X_train) - y_train)**2))
print ('Testing MSE: ', np.mean((regr.predict(X_test) - y_test)**2))
regr_lasso = linear_model.Lasso(alpha = .3)
regr_lasso.fit(X_train , y_train)
print ('Coeff and intercept:',regr_lasso.coef_)
print ('Tesing Score:', regr_lasso.score(X_test ,y_test))
print ('Training MSE:', np.mean((regr_lasso.predict(X_train) - y_train)**2))
print ( 'Training MSE:',np.mean((regr_lasso.predict(X_test) - y_test)**2))
ind = np.argsort(np.abs(regr_lasso.coef_))
print ('Ordered variable (from less to more important):',boston.feature_names[ind])
import sklearn.feature_selection as fs
selector = fs.SelectKBest(score_func = fs.f_regression ,k = 5)
selector.fit_transform(X_train , y_train) 
selector.fit(X_train ,y_train)
print ('Selected features:', zip(selector.get_support(), boston.feature_names))
import matplotlib.pyplot as plt
clf = LinearRegression()
clf.fit(boston.data , boston.target)
predicted = clf.predict(boston.data)
plt.scatter(boston.target , predicted , alpha = 0.3)
plt.plot([0, 50], [0, 50], '--k')
plt.axis('tight')
plt.xlabel('True price ($1000s)')
plt.ylabel('Predicted price ($1000s)')

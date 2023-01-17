# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import seaborn as sn
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#首先在excel中将所有数据中溢出单元1.79769313e+308全部清空,再进行数据载入
#03年1387行
data_03 = pd.read_excel('../input/GLAS_Landsat_2003.xlsx')
#04年1084行
data_04 = pd.read_excel('../input/GLAS_Landsat_2004.xlsx')
#05年1020行
data_05 = pd.read_excel('../input/GLAS_Landsat_2005.xlsx')
#06年1404行
data_06 = pd.read_excel('../input/GLAS_Landsat_2006.xlsx')
data_03.head()
#去掉03年莫名其妙多出来的第30列
data_03.drop(data_03.columns[30],axis=1,inplace=True) 
#去掉无用列
useless_columns = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19, 21,22,23]
data_03.drop(data_04.columns[useless_columns],axis=1,inplace=True) 
data_04.drop(data_04.columns[useless_columns],axis=1,inplace=True) 
data_05.drop(data_05.columns[useless_columns],axis=1,inplace=True)
data_06.drop(data_06.columns[useless_columns],axis=1,inplace=True) 

data_03.info()
means = data_03['b1_extract'].get_values().mean()
data_03['b1_extract'] = data_03['b1_extract'].apply(lambda x : means if (x>(1.5*means)) else x ).apply(lambda x: means if x<0.3*means else x)
#作出相关性图
plt.figure(figsize=(15,15))
corrDf=data_03.corr()
mask=np.array(corrDf)
mask[np.tril_indices_from(mask)]=False
plt.subplot(2,2,1)
sn.heatmap(corrDf,mask=mask,annot=True,square=True)


corrDf=data_04.corr()
mask=np.array(corrDf)
mask[np.tril_indices_from(mask)]=False
plt.subplot(2,2,2)
sn.heatmap(corrDf,mask=mask,annot=True,square=True)

corrDf=data_05.corr()
mask=np.array(corrDf)
mask[np.tril_indices_from(mask)]=False
plt.subplot(2,2,3)
sn.heatmap(corrDf,mask=mask,annot=True,square=True)


corrDf=data_06.corr()
mask=np.array(corrDf)
mask[np.tril_indices_from(mask)]=False
plt.subplot(2,2,4)
sn.heatmap(corrDf,mask=mask,annot=True,square=True)


plt.show()
def data_split(df, part1_percent=0.8, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df)
    part1_end = int(part1_percent * m)
    part1 = df.iloc[perm[:part1_end]]
    part2 = df.iloc[perm[part1_end:]]
    return part1, part2
train_valid_03, test_03 = data_split(data_03)
train_valid_04, test_04 = data_split(data_04)
train_valid_05, test_05 = data_split(data_05)
train_valid_06, test_06 = data_split(data_06)
train_valid_data = pd.concat([train_valid_03, train_valid_04, train_valid_05, train_valid_06])
plt.figure(figsize=(15,15))
corrDf=train_valid_data.corr()
mask=np.array(corrDf)
mask[np.tril_indices_from(mask)]=False
plt.subplot(2,2,1)
sn.heatmap(corrDf,mask=mask,annot=True,square=True)
plt.show()
train_valid_x = train_valid_data.drop(['Tree_Heigh'], axis=1)
train_valid_y = train_valid_data['Tree_Heigh']
#train_valid_x = train_valid_data.drop(['map_y','Tree_Heigh'], axis=1)
#train_valid_y = train_valid_data['map_y']
from sklearn.cross_validation import KFold
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
ridge_regression = Ridge()

parameters = {'alpha': [100, 20, 10, 7, 5, 3, 1, 0.5, 0.1, 0.005, 0.001, 0.0005, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0],
              'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
             }
grid = GridSearchCV(ridge_regression, parameters, cv=5)
grid.fit(train_valid_x,train_valid_y)
clf = grid.best_estimator_
print(clf)
clf.fit(train_valid_x, train_valid_y)
acc_ridge = clf.score(test_03.drop(['Tree_Heigh'], axis=1), test_03['Tree_Heigh'])
acc_ridge
"""
svr = SVR()
parameters = {'C':[10, 1, 0.1, 0.01, 0.01, 0.001, 0.0001],
              'epsilon':[1, 0.1, 0.001, 0.0001, 0.00001],
              'kernel':['poly', 'rbf', 'sigmoid'],
              'gamma':['scale']
             }
grid = GridSearchCV(svr, parameters, cv=5)
grid.fit(train_valid_x,train_valid_y['Tree_Heigh'])
clf = grid.best_estimator_
print(clf)
"""
#以下是得出的最优模型, 但是r^2还是接近于0
clf = SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.001, gamma='scale',
  kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
clf.fit(train_valid_x, train_valid_y)
acc_svr = clf.score(test_03.drop(['Tree_Heigh'], axis=1), test_03['Tree_Heigh'])
acc_svr
#接下来求rmse
from sklearn.metrics import mean_squared_error
yp = clf.predict(test_03.drop(['Tree_Heigh'], axis=1))
rmse = mean_squared_error(test_03['Tree_Heigh'],yp)
rmse
#以下是得出的最优模型, 但是r^2还是接近于0
clf = SVR(C=1, cache_size=200, coef0=0.0, degree=3, epsilon=0.001, gamma='scale',
  kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
clf.fit(train_valid_x[['b2_extract', 'b4_extract', 'b5_extract']], train_valid_y)
acc_svr = clf.score(test_03[['b2_extract', 'b4_extract', 'b5_extract']], test_03['Tree_Heigh'])
acc_svr

test_test = train_valid_03.sample(frac=0.3)
clf = SVR(C=1, cache_size=200, coef0=0.0, degree=3, epsilon=0.001, gamma='scale',
  kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
clf.fit(train_valid_03[['b2_extract', 'b4_extract', 'b5_extract']], train_valid_03['Tree_Heigh'])
acc_svr = clf.score(test_test[['b2_extract', 'b4_extract', 'b5_extract']], test_test['Tree_Heigh'])
acc_svr

train_valid_03.head()
data_03.plot.scatter(x=['b1_extract'],y=['Tree_Heigh'])
data_03['map_y'] = np.log(data_03['Tree_Heigh'])
data_03['map_x'] = data_03['b1_extract'] ** -5
data_03.head()
plt.figure(figsize=[20,10])
plt.scatter(data_03['map_x'],data_03['map_y'])
plt.xlim((0,40))

data_03['map_y'] = data_03['Tree_Heigh']** -2
train_03,test_03 = data_split(data_03)
train_03_x = train_03.drop(['map_y','Tree_Heigh'], axis=1)
train_03_y = train_03['map_y']
test_03_x = test_03.drop(['map_y','Tree_Heigh'], axis=1)
test_03_y = test_03['map_y']

from sklearn.linear_model import LassoCV
lasso = LassoCV()
lasso.fit(train_03_x,train_03_y)
predict_03_x = lasso.predict(test_03_x)

lasso.score(test_03_x,test_03_y)
predict_03_x**(-1/2)
test_03_y.get_values()
plt.figure()
plt.scatter(x=test_03_y.get_values()** (-1/2), y=predict_03_x** (-1/2))


# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
%matplotlib inline



import matplotlib.pyplot as plt



plt.rcParams['image.cmap'] = 'bwr' 



from pandas.tools.plotting import scatter_matrix
df = pd.read_csv("../input/train.csv")
df.describe()
df['SalePrice'].plot(kind='hist')
plt.figure(figsize=(20,20))

corr = df.corr()

plt.matshow(corr,fignum=0, vmin=-1, vmax=1)

plt.xticks(np.arange(len(corr.columns)),corr.columns,rotation='vertical')

plt.yticks(np.arange(len(corr.columns)),corr.columns)

plt.colorbar()
corr.columns
objcolumns = list(filter(lambda col:df[col].dtype == np.object, df.columns))

numcolumns = list(filter(lambda col:df[col].dtype != np.object, df.columns))
for col in objcolumns:

    print(col, df[col].unique())
for col in numcolumns:

    print(col, df[col].unique().size)
newdf = df[numcolumns].join(pd.get_dummies(df[objcolumns])).fillna(0)
newdf.describe()
#plt.style.use('grayscale')

plt.figure(figsize=(30,30))

corr = newdf.corr()

plt.matshow(corr,fignum=0,vmin=-1,vmax=1)

plt.xticks(np.arange(len(corr.columns)),corr.columns,rotation='vertical')

plt.yticks(np.arange(len(corr.columns)),corr.columns)

plt.colorbar()
pca_inputs = list(filter(lambda x: x!='SalePrice' and x!='Id', newdf.columns))
len(pca_inputs)
from sklearn.decomposition import PCA
pca = PCA(n_components=30) 

pca.fit(newdf[pca_inputs])
plt.figure(figsize=(100,30))

plt.matshow(pca.components_.transpose(),fignum=0)

plt.yticks(np.arange(len(pca_inputs)),pca_inputs)

plt.colorbar()
pcadf = newdf[['SalePrice']].copy()

pccol = pca.transform(newdf[pca_inputs])
for icol in range(pccol.shape[1]):

    pcadf['Comp%d' % icol] = pccol[:,icol]
#plt.style.use('grayscale')

plt.figure(figsize=(30,30))

corr = pcadf.corr()

plt.matshow(corr,fignum=0)

plt.xticks(np.arange(len(corr.columns)),corr.columns,rotation='vertical')

plt.yticks(np.arange(len(corr.columns)),corr.columns)

plt.colorbar()
corr.columns
corr['rank'] = np.abs(corr['SalePrice'])
sort_corr = corr.sort_values(by=['rank'],ascending=False)

sort_corr
selected_comps = sort_corr.index[1:15]
selected_comps
ax = scatter_matrix(pcadf[['SalePrice']+list(selected_comps)],figsize=[20,20],c=np.floor(np.log(df['SalePrice'])*10).astype(np.int32))
from sklearn import ensemble
from sklearn import cross_validation



X, y = pcadf[selected_comps], pcadf['SalePrice']





X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y)
from sklearn.metrics import mean_squared_error

from math import sqrt



params = {'n_estimators': 1000, 'max_depth': 30, 'min_samples_split': 10,

          'learning_rate': 0.005, 'loss': 'ls'}

clf = ensemble.GradientBoostingRegressor(**params)



clf.fit(X_train, y_train)

mse = mean_squared_error(np.log(y_test), np.log(clf.predict(X_test)))

print("RMSE: %.4f" % sqrt(mse))
# compute test set deviance

test_score = np.zeros((params['n_estimators'],), dtype=np.float64)



for i, y_pred in enumerate(clf.staged_predict(X_test)):

    test_score[i] = clf.loss_(y_test, y_pred)



plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)

plt.title('Deviance')

plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',

         label='Training Set Deviance')

plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',

         label='Test Set Deviance')

plt.legend(loc='upper right')

plt.xlabel('Boosting Iterations')

plt.ylabel('Deviance')
from sklearn.feature_selection import SelectKBest, mutual_info_regression
select = SelectKBest(score_func=mutual_info_regression,k=15)



select.fit(pccol,y)
kbest_comps = list(map(lambda x: 'Comp%d' % (x[0]), filter(lambda y: y[1], enumerate(select.get_support()))))
kbest_comps
from sklearn import cross_validation



X, y = pcadf[kbest_comps], pcadf['SalePrice']





X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y)
from sklearn.metrics import mean_squared_error

from math import sqrt



params = {'n_estimators': 1000, 'max_depth': 30, 'min_samples_split': 10,

          'learning_rate': 0.005, 'loss': 'ls'}

clf = ensemble.GradientBoostingRegressor(**params)



clf.fit(X_train, y_train)

mse = mean_squared_error(np.log(y_test), np.log(clf.predict(X_test)))

print("RMSE: %.4f" % sqrt(mse))
# compute test set deviance

test_score = np.zeros((params['n_estimators'],), dtype=np.float64)



for i, y_pred in enumerate(clf.staged_predict(X_test)):

    test_score[i] = clf.loss_(y_test, y_pred)



plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)

plt.title('Deviance')

plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',

         label='Training Set Deviance')

plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',

         label='Test Set Deviance')

plt.legend(loc='upper right')

plt.xlabel('Boosting Iterations')

plt.ylabel('Deviance')
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



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.linear_model import Ridge, Lasso,ElasticNet, ElasticNetCV

from sklearn.metrics import r2_score
auto=pd.read_csv('../input/autompg-dataset/auto-mpg.csv')

auto.head()
auto.shape
auto.info()
auto.isna().sum()
auto.describe()
auto['horsepower']=auto['horsepower'].replace('?',np.nan)
auto['horsepower']=auto['horsepower'].astype('float')
auto['horsepower']=auto['horsepower'].fillna(auto['horsepower'].median())
auto_final=auto.drop('car name',axis=1)
sns.distplot(auto_final['mpg'],kde=True)
print('The skewness of mpg: ',auto_final['mpg'].skew())

print('The kurtosis of mpg: ',auto_final['mpg'].kurt())
fig, ax = plt.subplots(6, 2, figsize = (15, 13))

sns.boxplot(x= auto_final["mpg"], ax = ax[0,0])

sns.distplot(auto_final['mpg'], ax = ax[0,1])



sns.boxplot(x= auto_final["cylinders"], ax = ax[1,0])

sns.distplot(auto_final['cylinders'], ax = ax[1,1])



sns.boxplot(x= auto_final["displacement"], ax = ax[2,0])

sns.distplot(auto_final['displacement'], ax = ax[2,1])



sns.boxplot(x= auto_final["horsepower"], ax = ax[3,0])

sns.distplot(auto_final['horsepower'], ax = ax[3,1])



sns.boxplot(x=auto_final["weight"], ax = ax[4,0])

sns.distplot(auto_final['weight'], ax = ax[4,1])



sns.boxplot(x= auto_final["acceleration"], ax = ax[5,0])

sns.distplot(auto_final['acceleration'], ax = ax[5,1])



plt.show()
sns.pairplot(auto_final,diag_kind='kde')
sns.heatmap(auto_final.corr(),annot=True)
auto_final=auto_final.transform(lambda x: x**0.5)
auto_final.head()
X=auto_final.drop('mpg',axis=1)

y=auto_final['mpg']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.30, random_state = 1) 



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
lin_reg = LinearRegression()

model = lin_reg.fit(X_train,y_train)

y_pred = lin_reg.predict(X_test)

print('Coefficients: ',lin_reg.coef_)

print('Intercept: ',lin_reg.intercept_)

print('Mean absolute error for test: ',mean_absolute_error(y_test,y_pred))

print('Mean Squared error for test: ',mean_squared_error(y_test,y_pred))

print('Root mean squared error for test: ',np.sqrt(mean_squared_error(y_test,y_pred)))

print('R^2 score for train: ',lin_reg.score(X_train, y_train))

print('R^2 score for test: ',lin_reg.score(X_test, y_test))
rr=Ridge(alpha=0.01, fit_intercept=True, normalize=False, max_iter=100,random_state=1)

rr.fit(X_train,y_train)
print('Coefficients: ',rr.coef_)

print('Intercept: ',rr.intercept_)

print('Score for train: ',rr.score(X_train,y_train))

print('Score for test: ',rr.score(X_test,y_test))
ll=Lasso(alpha=0.01, fit_intercept=True, normalize=False, max_iter=100, random_state=1)

ll.fit(X_train,y_train)
print('Coefficients: ',ll.coef_)

print('Intercept: ',ll.intercept_)

print('Score for train: ',ll.score(X_train,y_train))

print('Score for test: ',ll.score(X_test,y_test))
en_cv= ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, .995, 1], eps=0.001, n_alphas=100, fit_intercept=True, 

                        normalize=True, precompute='auto', max_iter=2000, tol=0.0001, cv=5, 

                        copy_X=True, verbose=0, n_jobs=-1, positive=False, random_state=1, selection='cyclic')

en_cv.fit(X_train,y_train)
print('Optimal alpha: ',en_cv.alpha_)

print('Optimal l1_ratio: ',en_cv.l1_ratio_)

print('Number of iterations: ',en_cv.n_iter_)
model = ElasticNet(l1_ratio=en_cv.l1_ratio_, alpha = en_cv.alpha_, max_iter=en_cv.n_iter_, fit_intercept=True, normalize = True)

model.fit(X_train, y_train)
print('Score for train: ',model.score(X_train,y_train))

print('Score for test: ',model.score(X_test,y_test))
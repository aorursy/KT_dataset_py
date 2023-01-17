# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
%matplotlib inline
boston=pd.read_csv('../input/housing.csv',delim_whitespace=True,header=None)
boston.shape
boston.info()
boston.describe()
col_name = ['CRIM', 'ZN' , 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
boston.columns=col_name
boston.head()
boston.isnull().sum()
sns.pairplot(boston)
corr_bost=boston.corr()
sns.heatmap(corr_bost,cmap='Reds')
corr_bost[(corr_bost > 0.6) | (corr_bost < -0.6 ) & (corr_bost != 1.00)].stack().sort_values().reset_index()
from scipy import stats
sns.distplot(boston['MEDV'], hist=True);
fig=plt.figure()
stats.probplot(boston['MEDV'],plot=plt);
X=boston.drop('MEDV',axis=1)
Y=boston['MEDV']
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=42)
X_test.shape
lr = LinearRegression()
lr.fit(X_train, y_train)
lr.coef_ , lr.intercept_
lr.score(X_train,y_train)
lr.score(X_test,y_test)
y_train_pred = lr.predict(X_train)
y_test_predict=lr.predict(X_test)
residual_test=((y_test_predict-y_test)**2).sum()
np.sqrt(residual_test/X.shape[0])
from sklearn.metrics import r2_score
r2_score(y_test,y_test_predict)
mean_squared_error(y_test, y_test_predict)
mean_squared_error(y_train,y_train_pred)
kfold = KFold(n_splits=10, random_state=42)
score_cv = cross_val_score(lr,X_train,y_train,scoring='neg_mean_squared_error',cv=kfold)
score_cv.shape
from sklearn.model_selection import cross_val_predict
y_pred_cv = cross_val_predict(lr,X_train,y_train,cv=kfold)
y_pred_cv.shape,X_train.shape
r2_score(y_train,y_pred_cv)
mean_squared_error(y_train, y_pred_cv)
y_pred__test_cv = cross_val_predict(lr,X_test,y_test,cv=151)
mean_squared_error(y_test, y_pred__test_cv)
r2_score(y_test, y_pred__test_cv)
boston.head()
bins = [0, 1, 5, 10, 25, 50, 100]
### to bin the values into different bins
pd.cut(boston['CRIM'],bins=bins).value_counts().plot.bar()
boston['CHAS'].value_counts().plot.bar();
boston.plot(kind='line',subplots=True,layout=(7,2),figsize=(20,20));
boston.plot(x='TAX',y='RAD',kind='scatter')#,subplots=True,layout=(7,2),figsize=(20,20));
boston.corr()['INDUS']['RAD']
boston[['RAD','TAX','INDUS']]
plt.plot(y_train.values,y_train_pred,kind='scatter');
plt.plot(y_train.values, 'r^')
plt.plot(y_train_pred, 'go') 
#plt.plot(y_pred_cv,'g')
plt.plot(y_train.values-y_train_pred)

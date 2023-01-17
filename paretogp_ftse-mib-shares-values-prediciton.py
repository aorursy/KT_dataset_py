# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn as sk
import matplotlib
from matplotlib import pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVR
from sklearn.ensemble import AdaBoostRegressor
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_out=pd.read_csv('/kaggle/input/ftse-mib-stocks/FTSE_MIB_Main_Stocks.csv')
class Upd_Data(BaseEstimator, TransformerMixin):
    def __init__ (self, add_Adj_Close=True):
        self.add_Adj_Close=add_Adj_Close
       
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        
        if(self.add_Adj_Close):
            return np.array(X.drop(['Year','Shares'],axis=1))
        else:
            return np.array(X)
res_dict=dict()
df_out.info()
df=df_out.drop(['Awards_issued','Investment_Profits','Other_revenues','Profit_for_the_year','Total_Loans','Interest_margin','Brokerage_margin','Management:result','Total_assets'],axis=1)
df=df.dropna()
df.info()
num_p=Pipeline([('attr_add',Upd_Data(add_Adj_Close=True)),('std_scaler',StandardScaler())],)
df_tf=pd.DataFrame(num_p.fit_transform(df),columns=df.columns[0:14].append(pd.Index(['Incr_prc'])))
X=np.array(df_tf.drop(['Ann.Increase\[%\]'],axis=1))
y=np.array(df_tf['Ann.Increase\[%\]'])
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
plt.plot(y)


plt.axhline(y.mean(), color='g')
plt.axhline(np.percentile(y, 75), color='#d62728')
plt.grid()

lin_reg=LinearRegression()
scores=cross_val_score(lin_reg,X_tr,y_tr,scoring='neg_mean_squared_error',cv=10)
rmse_lin_reg=np.sqrt(-scores)
print(rmse_lin_reg)
res_dict['Lin_Reg']=rmse_lin_reg.mean()
f_reg=RandomForestRegressor(n_estimators=30)
param_grid=[{'n_estimators':[30,40,50],},]
grid_search=GridSearchCV(f_reg,param_grid,cv=5,scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(X_tr,y_tr)
print(grid_search.best_estimator_)
scores=cross_val_score(f_reg,X_tr,y_tr,scoring='neg_mean_squared_error',cv=10)
rmse_for_reg=np.sqrt(-scores)
print(rmse_for_reg)
res_dict['Rand_For_Reg']=rmse_for_reg.mean()
eln_reg=ElasticNet()
param_grid=[{'alpha':[0.1,0.2,0.5,1],'l1_ratio':[0.01,0.1,0.5,1],'max_iter':[2000000000],},]
grid_search=GridSearchCV(eln_reg,param_grid,cv=5,scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(X_tr,y_tr)
grid_search.best_estimator_
eln_reg=ElasticNet(alpha=0.1, max_iter=2000000000)
scores=cross_val_score(eln_reg,X_tr,y_tr,scoring='neg_mean_squared_error',cv=10)
rmse_eln_reg=np.sqrt(-scores)
res_dict['El_Net_Reg']=rmse_eln_reg.mean()
lasso_reg=Lasso(alpha=0.1,fit_intercept=False, tol=0.00000000000001,
          max_iter=2000000000, positive=True)
scores=cross_val_score(lasso_reg,X_tr,y_tr,scoring='neg_mean_squared_error',cv=10)
rmse_lasso_reg=np.sqrt(-scores)
res_dict['Lasso_Reg']=rmse_lasso_reg.mean()
svm_reg=LinearSVR()
param_grid=[{'epsilon':[0.01,0.05,0.1],'max_iter':[2000000000],},]
grid_search=GridSearchCV(svm_reg,param_grid,cv=5,scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(X_tr,y_tr)
grid_search.best_estimator_
svm_reg=LinearSVR(epsilon=0.1, max_iter=2000000000)
scores=cross_val_score(svm_reg,X_tr,y_tr,scoring='neg_mean_squared_error',cv=10)
rmse_svm_reg=np.sqrt(-scores)
res_dict['SVM_Reg']=rmse_svm_reg.mean()
ada_reg=AdaBoostRegressor(base_estimator=ElasticNet(alpha=0.1, max_iter=2000000000))
param_grid=[{'n_estimators':[100,110,120],'learning_rate':[1],},]
grid_search=GridSearchCV(ada_reg,param_grid,cv=5,scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(X_tr,y_tr)
grid_search.best_estimator_
ada_eln_reg=AdaBoostRegressor(base_estimator=ElasticNet(alpha=0.1, max_iter=2000000000),
                  learning_rate=1, n_estimators=120)
scores=cross_val_score(ada_eln_reg,X_tr,y_tr,scoring='neg_mean_squared_error',cv=10)
rmse_ada_eln_reg=np.sqrt(-scores)
res_dict['ADA_ELN_Reg']=rmse_ada_eln_reg.mean()
print(res_dict)
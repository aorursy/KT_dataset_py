import datetime           # for start and end periods
import time
import pandas as pd
import sklearn as sk
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
%matplotlib inline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import LinearSVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
res_dict=dict()
class Yhoodwn(object):
    def __init__(self, interval="1d"):
        self.url = "https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={start_time}&period2={end_time}&interval={interval}&events=history"
        self.interval = interval
    def __build_url(self, ticker, start_date, end_date):
        return self.url.format(ticker=ticker, start_time=start_date, end_time=end_date, interval=self.interval)
    def get_data(self, ticker, start_date, end_date):
        # must pass datetime into this function
        epoch_start = int(time.mktime(start_date.timetuple()))
        epoch_end = int(time.mktime(end_date.timetuple()))
        return pd.read_csv(self.__build_url(ticker, epoch_start, epoch_end))

class Upd_Data(BaseEstimator, TransformerMixin):
    def __init__ (self, add_Adj_Close=True,N=5,G=1):
        self.add_Adj_Close=add_Adj_Close
        self.N=N
        self.G=G
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        buf0=(X.High-X.Low)/2+X.Low
        buf1=np.convolve(buf0, np.ones((self.N,))/self.N, mode='valid')
        
        MM=np.append(buf1,np.ones(len(buf0)-len(buf1))*buf1[len(buf1)-1])
        
        X['Gradient']=np.gradient(MM,1)*self.G
        if(self.add_Adj_Close):
            return np.array(X.drop(['Date'],axis=1))
        else:
            return np.array(X.drop(['Date','Adj Close','Volume'],axis=1))
dh = Yhoodwn()
now = datetime.datetime(2020, 9, 14)    # get data up to 
then = datetime.datetime(2000, 1, 1)        # get data from
df = dh.get_data("EXO.MI", then, now)
    
print(df)
df.info()
df.head()
num_p=Pipeline([('attr_add',Upd_Data(add_Adj_Close=True,N=2,G=1)),('imputer',SimpleImputer(strategy='median')),('std_scaler',StandardScaler())],)
df_tf=pd.DataFrame(num_p.fit_transform(df),columns=df.columns[1:7].append(pd.Index(['Gradient'])))
df_tf.describe()
df_tf.plot()
df_tf['Date']=pd.to_datetime(df.Date)
df_tf.info()
X=np.array(df_tf.drop(['Date','Low'],axis=1))[:-1]
y=np.array(df_tf.Low)[1:]
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
lin_reg=LinearRegression()
scores=cross_val_score(lin_reg,X_tr,y_tr,scoring='neg_mean_squared_error',cv=10)
rmse_lin_reg=np.sqrt(-scores)
print(rmse_lin_reg)
res_dict['Lin_Reg']=rmse_lin_reg
f_reg=RandomForestRegressor(n_estimators=40)
scores=cross_val_score(f_reg,X_tr,y_tr,scoring='neg_mean_squared_error',cv=10)
rmse_for_reg=np.sqrt(-scores)
print(rmse_for_reg)
res_dict['Rand_For_Reg']=rmse_for_reg
param_grid=[{'n_estimators':[30,40,50],},]
grid_search=GridSearchCV(f_reg,param_grid,cv=5,scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(X_tr,y_tr)
grid_search.best_estimator_
cvres=grid_search.cv_results_
for mean_score,params in zip(cvres['mean_test_score'],cvres['params']):
    print(np.sqrt(-mean_score),params)
ridge_reg=Ridge(alpha=1,solver='cholesky')
scores=cross_val_score(ridge_reg,X_tr,y_tr,scoring='neg_mean_squared_error',cv=10)
rmse_ridge_reg=np.sqrt(-scores)
print(rmse_ridge_reg)
print((rmse_lin_reg-rmse_ridge_reg).mean())
res_dict['Ridge_Reg']=rmse_ridge_reg
lasso_reg=Lasso(alpha=0.1,fit_intercept=False, tol=0.00000000000001,
          max_iter=100000000, positive=True)
scores=cross_val_score(lasso_reg,X_tr,y_tr,scoring='neg_mean_squared_error',cv=10)
rmse_lasso_reg=np.sqrt(-scores)
print(rmse_lasso_reg)
print((rmse_lin_reg-rmse_lasso_reg).mean())
res_dict['Lasso_Reg']=rmse_lasso_reg
eln_reg=ElasticNet(alpha=0.1,l1_ratio=0.1)
scores=cross_val_score(eln_reg,X_tr,y_tr,scoring='neg_mean_squared_error',cv=10)
rmse_eln_reg=np.sqrt(-scores)
print(rmse_eln_reg)
print((rmse_lin_reg-rmse_eln_reg).mean())
param_grid=[{'alpha':[0.1,0.2,0.5,1],'l1_ratio':[0.1,0.2,0.5,1],'max_iter':[100000000],},]
grid_search=GridSearchCV(eln_reg,param_grid,cv=5,scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(X_tr,y_tr)
grid_search.best_estimator_
cvres=grid_search.cv_results_
for mean_score,params in zip(cvres['mean_test_score'],cvres['params']):
    print(np.sqrt(-mean_score),params)
res_dict['El_Net_Reg']=rmse_eln_reg
svm_reg=LinearSVR()
param_grid=[{'epsilon':[0.01,0.05,0.1],'max_iter':[100000000],},]
grid_search=GridSearchCV(svm_reg,param_grid,cv=5,scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(X_tr,y_tr)
grid_search.best_estimator_
cvres=grid_search.cv_results_
for mean_score,params in zip(cvres['mean_test_score'],cvres['params']):
    print(np.sqrt(-mean_score),params)
svm_reg=LinearSVR(epsilon=0.05, max_iter=1000000000)
scores=cross_val_score(svm_reg,X_tr,y_tr,scoring='neg_mean_squared_error',cv=10)
rmse_svm_reg=np.sqrt(-scores)
print(rmse_svm_reg)
print((rmse_lin_reg-rmse_svm_reg).mean())
res_dict['SVM_Reg']=rmse_svm_reg
ada_reg=AdaBoostRegressor(base_estimator=Ridge(alpha=1,solver='cholesky'),
                  learning_rate=1, n_estimators=110)
param_grid=[{'n_estimators':[100,110,120],'learning_rate':[1],},]
grid_search=GridSearchCV(ada_reg,param_grid,cv=5,scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(X_tr,y_tr)
grid_search.best_estimator_
cvres=grid_search.cv_results_
for mean_score,params in zip(cvres['mean_test_score'],cvres['params']):
    print(np.sqrt(-mean_score),params)
res_dict['ADA_SVR_Reg']=0.06570451127388993
ada_svr_reg=AdaBoostRegressor(base_estimator=LinearSVR(epsilon=0.01, max_iter=1000000000),learning_rate=1, n_estimators=100)
scores=cross_val_score(ada_svr_reg,X_tr,y_tr,scoring='neg_mean_squared_error',cv=10)
rmse_ada_svr_reg=np.sqrt(-scores)
print(rmse_ada_svr_reg)
print((rmse_lin_reg-rmse_ada_svr_reg).mean())
res_dict['ADA_SVR_Reg']=rmse_ada_svr_reg
ada_eln_reg=AdaBoostRegressor(base_estimator=ElasticNet(alpha=0.1,l1_ratio=0.1),learning_rate=1, n_estimators=120)
scores=cross_val_score(ada_eln_reg,X_tr,y_tr,scoring='neg_mean_squared_error',cv=10)
rmse_ada_eln_reg=np.sqrt(-scores)
print(rmse_ada_eln_reg)
print((rmse_lin_reg-rmse_ada_eln_reg).mean())
res_dict['ADA_ELN_Reg']=rmse_ada_eln_reg
gbrt=GradientBoostingRegressor()
param_grid=[{'max_depth':[2],'n_estimators':[120,130],'learning_rate':[0.1,0.5],},]
grid_search=GridSearchCV(gbrt,param_grid,cv=5,scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(X_tr,y_tr)
grid_search.best_estimator_
cvres=grid_search.cv_results_
for mean_score,params in zip(cvres['mean_test_score'],cvres['params']):
    print(np.sqrt(-mean_score),params)
gbrt=GradientBoostingRegressor(learning_rate= 0.1,max_depth=2, n_estimators=120)
scores=cross_val_score(gbrt,X_tr,y_tr,scoring='neg_mean_squared_error',cv=10)
rmse_gbrt_reg=np.sqrt(-scores)
print(rmse_gbrt_reg)
print((rmse_lin_reg-rmse_gbrt_reg).mean())
res_dict['GBRT']=rmse_gbrt_reg
for i in res_dict:
    print(i+':')
    print(res_dict[i].mean())

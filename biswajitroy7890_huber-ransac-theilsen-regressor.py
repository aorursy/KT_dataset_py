import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True)
pd.set_option('display.max_columns',8000)
pd.set_option('display.max_rows',7000)
from google.colab import files
files.upload()
dftrain=pd.read_csv('df_train.csv')
dftest=pd.read_csv('df_test.csv')
dftrain.head()
dftest.head()
column_object=['MSZoning','Street','LotShape','LandContour','LotConfig','LandSlope','Neighborhood','Condition1','Condition2',
 'BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond',
 'BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','GarageType','GarageFinish',
 'GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition','FireplaceQu']
dftest.head()
dftrain.shape
dftest.shape
except_columns=[]
for col in dftrain.columns:
  if(col not in column_object):
    except_columns.append(col)
  else:
      print(col) 
from sklearn.preprocessing import OrdinalEncoder
enc=OrdinalEncoder()
X_train_feautres=enc.fit_transform(dftrain[column_object])
df_train1=pd.DataFrame(X_train_feautres, columns=column_object)
df_train1.shape
dftrain=dftrain.drop(labels=column_object, axis=1)
dftrain.shape
df_train1.head()
frames=[dftrain,df_train1]
df_train = pd.concat(frames,axis=1)
df_train.shape
X_test_feautres=enc.fit_transform(dftest[column_object])
df_test1=pd.DataFrame(X_test_feautres, columns=column_object)
df_test1.shape
dftest=dftest.drop(labels=column_object, axis=1)
dftest.shape
frames1=[dftest,df_test1]
df_test = pd.concat(frames1,axis=1)
df_test.shape
#############################################################################################################
print(df_train.shape)
print(df_test.shape)
df_train.head(2)
pred_list=[]
for col in df_train.columns:
   corr_point=df_train['SalePrice'].corr(df_train[col],method='pearson',min_periods=1)
   if(corr_point>=0.5):
     pred_list.append(col)
     print(col)
   else:
     pass  
pred_list
pred_list.remove('SalePrice')
#Target=df_train['SalePrice'].values
#df_train=df_train.drop(['SalePrice','Id'], axis=1)
selected_feat=df_train.columns
Feautres=df_train[selected_feat].values
y=Target
X=Feautres
y
X=np.log1p(X)
y=np.log1p(y)
print(X[:3])
print(y[:3])
from sklearn.preprocessing import StandardScaler,MinMaxScaler,normalize
pt_predictor=MinMaxScaler() 
pt_target=MinMaxScaler() 
X=pt_predictor.fit_transform(X)
y=pt_target.fit_transform(np.reshape(y,(-1,1)))
print(np.any(np.isnan(X)))
print(np.all(np.isfinite(X)))

from sklearn.linear_model import MultiTaskElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import train_test_split
import math
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
kf=KFold(n_splits=7, random_state=None, shuffle=False)
for train_index, test_index in kf.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    gb=xgb.XGBRegressor(n_estimators=2000,n_jobs=-1,learning_rate=2,max_depth=16,objective='reg:squarederror', min_child_weight=3,colsample_bytree=0.7)
    rf=RandomForestRegressor(n_estimators=2000, criterion='mse', max_depth=15)
    gb.fit(X_train,y_train)
    rf.fit(X_train,y_train)
    prediction1=gb.predict(X_test)
    prediction2=rf.predict(X_test)
    Y_orig=pt_target.inverse_transform(y_test)
    pred_orig1=pt_target.inverse_transform(np.reshape(prediction1,(-1,1)))
    pred_orig2=pt_target.inverse_transform(np.reshape(prediction2,(-1,1)))
    pred_final=(pred_orig1+pred_orig2)/2
    #Y_orig=np.exp(Y_orig)
    #pred_orig=np.exp(pred_orig)
    print('Accuracy',100-math.sqrt(metrics.mean_squared_error(Y_orig,pred_final)))


predictors=[]
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
estimator =RandomForestRegressor()
selector = RFE(estimator)
selector = selector.fit(X, y)
feature_importances = pd.Series(selector.ranking_, index=selected_feat)
feature_importances.nsmallest(15)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
kf=KFold(n_splits=7, random_state=None, shuffle=False)
for train_index, test_index in kf.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    kernel= DotProduct()+WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel,random_state=0,n_restarts_optimizer=6)
    gpr.fit(X_train,y_train)
    prediction=gpr.predict(X_test)
    print('Accuracy',100-math.sqrt(metrics.mean_squared_error(y_test,prediction)))


from sklearn.linear_model import HuberRegressor, RANSACRegressor,TheilSenRegressor
from sklearn.ensemble import RandomForestRegressor
kf=KFold(n_splits=7, random_state=None, shuffle=False)
for train_index, test_index in kf.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    huber = HuberRegressor(epsilon=45, max_iter=8000, alpha=0.00001)
    base_estimator=RandomForestRegressor()
    ransac=RANSACRegressor(base_estimator,max_trials=2000)
    theil=TheilSenRegressor(max_iter=2000,n_subsamples=75)
    model=theil.fit(X,y)
    prediction=model.predict(X_test)
    print('Accuracy',100-math.sqrt(metrics.mean_squared_error(y_test,prediction)))


#df_test=df_test.drop('Id', axis=1)
col_test=df_test.columns
X_test=df_test[selected_feat].values
X_test=np.log1p(X_test)
test_predictor=MinMaxScaler() 
X_Final=test_predictor.fit_transform(X_test)
pred_Test1=gb.predict(X_Final)
pred_Test2=rf.predict(X_Final)
pred_test_final=(pred_Test1+pred_Test2)/2

pred=pt_target.inverse_transform(np.reshape(pred_test_final,(-1,1)))
#pred=np.exp(predictions)
pred_ran=model.predict(X_test)
sample_df=pd.DataFrame(dftest, columns=['Id'])
sample_df['SalePrice']=pred_ran
sample_df.to_csv('sample_df.csv',index=False)
pred_ran[:10]
dftrain.hist('SalePrice',figsize=(15,5)) 
sample_df.hist('SalePrice',figsize=(15,5)) 
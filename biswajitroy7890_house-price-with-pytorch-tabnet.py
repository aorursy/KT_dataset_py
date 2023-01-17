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





Target=df_train['SalePrice'].values
df_train=df_train.drop(['SalePrice','Id'], axis=1)
selected_feat=df_train.columns
Feautres=df_train[selected_feat].values
y=Target
X=Feautres
X
X=np.log1p(X)
y=np.log1p(y)
print(X[:3])
print(y[:3])
from sklearn.preprocessing import StandardScaler,MinMaxScaler,normalize
pt_predictor=StandardScaler() 
pt_target=StandardScaler() 
X=pt_predictor.fit_transform(X)
y=pt_target.fit_transform(np.reshape(y,(-1,1)))
print(np.any(np.isnan(X)))
print(np.all(np.isfinite(X)))

from sklearn.linear_model import MultiTaskElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import train_test_split
import math
from sklearn.model_selection import KFold
kf=KFold(n_splits=7, random_state=None, shuffle=False)
for train_index, test_index in kf.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    gb=XGBRegressor(n_estimators=1200,n_jobs=-1,learning_rate=0.5,max_depth=6,objective='reg:squarederror',)
    rf=RandomForestRegressor(n_estimators=1200, criterion='mse', max_depth=5)
    gb.fit(X_train,y_train)
    rf.fit(X_train,y_train)
    prediction1=gb.predict(X_test)
    prediction2=rf.predict(X_test)
    Y_orig=pt_target.inverse_transform(y_test)
    pred_orig1=pt_target.inverse_transform(prediction1)
    pred_orig2=pt_target.inverse_transform(prediction2)
    pred_final=(pred_orig1+pred_orig2)/2
    #Y_orig=np.exp(Y_orig)
    #pred_orig=np.exp(pred_orig)
    print('Accuracy',100-math.sqrt(metrics.mean_squared_error(Y_orig,pred_final)))


params={
     'n_estimators':[100, 200,300,400],
      'max_depth':[3,4,5,6,]
   }
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
GV.score


#df_test=df_test.drop('Id', axis=1)
col_test=df_test.columns
X_test=df_test[selected_feat].values
X_test=np.log1p(X_test)
test_predictor=StandardScaler() 
X_Final=test_predictor.fit_transform(X_test)
pred_Test1=gb.predict(X_Final)
pred_Test2=rf.predict(X_Final)
pred_test_final=(pred_Test1+pred_Test2)/2

pred=pt_target.inverse_transform(np.reshape(pred_test_final,(-1,1)))
#pred=np.exp(predictions)
sample_df=pd.DataFrame(dftest, columns=['Id'])
sample_df['SalePrice']=pred
sample_df.to_csv('sample_df.csv',index=False)
pred[:10]
dftrain.hist('SalePrice',figsize=(15,5)) 
sample_df.hist('SalePrice',figsize=(15,5)) 
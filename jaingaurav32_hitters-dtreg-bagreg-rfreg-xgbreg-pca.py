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
Hitters=pd.read_csv('/kaggle/input/hitters/Hitters.csv')
Hitters.shape
Hitters.head()
Hitters.index.name='Players'
Hitters.head()
Hitters=Hitters.dropna()
Hitters.describe()
Hitters.Salary=np.log(Hitters.Salary)
import matplotlib.pyplot as plt

Hitters.hist('Salary')
Hitters=pd.get_dummies(Hitters,columns=['League','Division','NewLeague'],drop_first=True)

Hitters
X=Hitters.copy()

del X['Salary']

X.shape
y=Hitters.Salary

y.shape
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
from sklearn.tree import DecisionTreeRegressor
modelall=DecisionTreeRegressor(max_depth=4)
modelall.fit(X_train,y_train)

#initial max depth can be square root of number of features
modelall.score(X_train,y_train)
from sklearn.tree import export_graphviz

from IPython.display import Image

export_graphviz(modelall,out_file='modelall.dot',feature_names=X_train.columns)

! dot -Tpng modelall.dot -o modelall.png

Image('modelall.png')
y_pred_all=modelall.predict(X_test)
SSE_modelall=np.sum((y_pred_all-y_test)**2)

SSE_modelall
SST_modelall=np.sum((y_test-np.mean(y_train))**2)

SST_modelall
1-(SSE_modelall/SST_modelall)
from sklearn.metrics import mean_squared_error

RMSE_modelall=np.sqrt(mean_squared_error(y_pred_all,y_test))
RMSE_modelall
Parameters={'max_depth':[1,2,3,4,5,6]}
from sklearn.model_selection import GridSearchCV

grid=GridSearchCV(modelall,Parameters,cv=10,scoring='r2')
grid.fit(X_train,y_train)
grid.best_score_
grid.best_params_
model_prune=DecisionTreeRegressor(max_depth=3)
model_prune.fit(X_train,y_train)
model_prune.score(X_train,y_train)
y_pred_prune=model_prune.predict(X_test)
SSE_prune=np.sum((y_pred_prune-y_test)**2)

SSE_prune
SST_prune=np.sum((y_test-np.mean(y_train))**2)

SST_prune
R2_prune=1-(SSE_prune/SST_prune)

R2_prune
from sklearn.metrics import mean_squared_error

RMSE_prune=np.sqrt(mean_squared_error(y_test,y_pred_prune))

RMSE_prune
model_prune.feature_importances_
data=pd.Series(model_prune.feature_importances_,index=X_train.columns)

data.sort_values(ascending=True,inplace=True)

data.plot.barh()
export_graphviz(model_prune,out_file='model_prune.dot',feature_names=X_train.columns)

! dot -Tpng model_prune.dot -o model_prune.png

Image('model_prune.png')
from sklearn.ensemble import BaggingRegressor

model_bag=BaggingRegressor(base_estimator=DecisionTreeRegressor(max_depth=3), n_estimators=100, max_features=19, oob_score=True, random_state=42, verbose=2)
model_bag.fit(X_train,y_train)
model_bag.score(X_train,y_train)
model_bag.oob_score_
y_bag_pred=model_bag.predict(X_test)

y_bag_pred
SSE_bag=np.sum((y_bag_pred-y_test)**2)

SSE_bag
SST_bag=np.sum((y_test-np.mean(y_train))**2)

SST_bag
R2_bag=1-(SSE_bag/SST_bag)

R2_bag
RMSE_bag=np.sqrt(mean_squared_error(y_test,y_bag_pred))

RMSE_bag
from sklearn.ensemble import RandomForestRegressor
model_RF=RandomForestRegressor(n_estimators=200, max_depth=6, max_features=12, oob_score=True, random_state=42, verbose=2)
model_RF.fit(X_train,y_train)
model_RF.score(X_train,y_train)
model_RF.oob_score_
y_pred_RF=model_RF.predict(X_test)
SSE_RF=np.sum((y_pred_RF-y_test)**2)

SSE_RF
SST_RF=np.sum((y_test-np.mean(y_train))**2)

SST_RF
R2_RF=1-(SSE_RF/SST_RF)

R2_RF
RMSE_RF=np.sqrt(mean_squared_error(y_test,y_pred_RF))

RMSE_RF
model_RF.feature_importances_
dataRF=pd.Series(data=model_RF.feature_importances_, index=X_train.columns)

dataRF.sort_values(ascending=True, inplace=True)

dataRF.plot.barh()
ParametersRF={'max_depth':np.arange(4,7),'max_features':np.arange(10,15)}
gridRF=GridSearchCV(model_RF,ParametersRF,cv=5,)
gridRF.fit(X_train,y_train)
gridRF.best_params_
y_pred_RFtune=gridRF.predict(X_test)
RMSE_RFtune=np.sqrt(mean_squared_error(y_test,y_pred_RFtune))

RMSE_RFtune
#installing xgboost library and install it

#!pip install xgboost
import xgboost as xgb
model_xgb=xgb.XGBRegressor(objective='reg:linear',n_estimators=500)
model_xgb.fit(X_train,y_train)
model_xgb.score(X_train,y_train)
y_pred_xgb=model_xgb.predict(X_test)

y_pred_xgb
SSE_xgb=np.sum((y_pred_xgb-y_test)**2)

SSE_xgb
SST_xgb=np.sum((y_test-np.mean(y_train))**2)

SST_xgb
R2_xgb=1-(SSE_xgb/SST_xgb)

R2_xgb
RMSE_xgb=np.sqrt(mean_squared_error(y_test,y_pred_xgb))

RMSE_xgb
parameters_xgb={'max_depth':np.arange(1,4),'learning_rate':[0.1,0.01,0.001]}
grid_xgb=GridSearchCV(model_xgb,parameters_xgb)
grid_xgb.fit(X_train,y_train)
grid_xgb.best_score_
grid_xgb.best_params_
tune_xgb=xgb.XGBRegressor(objective='reg:linear',n_estimators=1000,max_depth=2,learning_rate=0.01)
tune_xgb.fit(X_train,y_train)
tune_xgb.score(X_train,y_train)
y_tune_xgb=tune_xgb.predict(X_test)
SSE_tune_xgb=np.sum((y_tune_xgb-y_test)**2)

SST_tune_xgb=np.sum((y_test-np.mean(y_train))**2)

R2_tune_xgb=1-(SSE_tune_xgb/SST_tune_xgb)

R2_tune_xgb
RMSE_tune_xgb=np.sqrt(mean_squared_error(y_test,y_tune_xgb))

RMSE_tune_xgb
from sklearn import linear_model
ridge_reg=linear_model.Ridge(alpha=0.1,normalize=True)

ridge_reg.fit(X_train,y_train)
ridge_reg.coef_
ridge_reg.score(X_train,y_train)
import random

np.random.seed=(42)

regCV=linear_model.RidgeCV(alphas=[0.0001,0.001,0.01,0.1,1,10,100,1000,10000],cv=5,normalize=True)
regCV.fit(X_train,y_train)
regCV.alpha_
ridgecv=linear_model.Ridge(alpha=1,normalize=True)
ridgecv.fit(X_train,y_train)
ridgecv.coef_
ridgecv.score(X_train,y_train)
y_predcv=ridgecv.predict(X_test)
sse_cv=np.sum((y_predcv-y_test)**2)

sst_cv=np.sum((y_test-np.mean(y_train))**2)

r2_cv=1-(sse_cv/sst_cv)

r2_cv
rmse_cv=np.sqrt(mean_squared_error(y_test,y_predcv))

rmse_cv
import matplotlib.pyplot as plt

import seaborn as sns
n_alphas=200

alphas=np.logspace(-1,4,n_alphas)

alphas[:10]
coef=[]

for a in alphas:

    ridge=linear_model.Ridge(alpha=a,normalize=True)

    ridge.fit(X_train,y_train)

    coef.append(ridge.coef_)

    

coef[:10]
datareg=pd.DataFrame(coef,columns=X_train.columns,index=alphas)

datareg['alphas']=datareg.index

datareg.head()
y_var=datareg.columns.difference(['alphas']) #it drops the alpha column from the column list

y_var
ax=plt.gca()

ax.set_xscale('log')

for i in range(0,16):

    ax=sns.lineplot(data=datareg,x='alphas',y=y_var[i])
reg_lasso=linear_model.Lasso(alpha=0.001,normalize=True)
reg_lasso.fit(X_train,y_train)
reg_lasso.score(X_train,y_train)
reg_lasso.coef_ #Lasso makes the sum of the coefficients zero hence it performs feature selection
coef=pd.Series(data=reg_lasso.coef_,index=X_train.columns)

coef.sort_values(ascending=True,inplace=True)

coef.plot.barh()
np.random.seed=(42)

reg_lassoCV=linear_model.LassoCV(alphas=[0.0001,0.001,0.01,0.1,1],max_iter=10000,cv=5,normalize=True)
reg_lassoCV.fit(X_train,y_train)
reg_lassoCV.alpha_
y_pred_lasso=reg_lassoCV.predict(X_test)
sse_lasso=np.sum((y_test-y_pred_lasso)**2)

sst_lasso=np.sum((y_test-np.mean(y_train))**2)

r2_lasso=1-sse_lasso/sst_lasso

r2_lasso
rmse_lasso=np.sqrt(mean_squared_error(y_test,y_pred_lasso))

rmse_lasso
nalpha=200

alphas=np.logspace(-4,-1,nalpha)

alphas[:10]
coef=[]

for a in alphas:

    reg=linear_model.Lasso(alpha=a,max_iter=10000,normalize=True)

    reg.fit(X_train,y_train)

    coef.append(reg.coef_)

    

coef[:10]
df_coef=pd.DataFrame(coef,index=alphas,columns=X_train.columns)
plt.figure(figsize=(10,10))

ax=plt.gca()

ax.plot(df_coef.index,df_coef.values)

ax.set_xscale('log')

plt.xlabel('alpha')

plt.ylabel('weights')

ax.get_ymajorticklabels()

plt.title('lasso regression as a function of the Regularization')

plt.axis('tight')

plt.legend(df_coef.columns)

plt.show()
from sklearn.preprocessing import scale

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
X_train.head()
scalar.fit(X_train) #fit will compute mean and standard deviation
scalar.transform(X_train) #transform will standardize the data
Scale_X=pd.DataFrame(scalar.transform(X_train),columns=X_train.columns)

Scale_X.head()
Scale_X.mean(),Scale_X.std()
pca_hitters=PCA()
pca_hitters.fit(Scale_X)
pca_hitters.n_components_
pca_hitters.components_ #Φ values
pca_loadings=pd.DataFrame(pca_hitters.components_,columns=X_train.columns)

pca_loadings.T
pca_hitters.explained_variance_
sum(pca_hitters.explained_variance_)
7.39/19.00
pca_hitters.explained_variance_ratio_
sum(pca_hitters.explained_variance_ratio_)
PC_score=pd.DataFrame(pca_hitters.fit_transform(Scale_X),columns=pca_loadings.T.columns)

PC_score
#plt.figure(figsize=(7,5))

#plt.plot(np.arange(1,20),pca_hitters.explained_variance_ratio_,'-o',label='Individual Components')

plt.plot(np.arange(1,20),np.cumsum(pca_hitters.explained_variance_ratio_),'-s',label='Cumulative')



plt.ylabel('Proportion of Varience Explained')

plt.xlabel('Principal Component')



#plt.xlim(0.75,4.25)

#plt.ylim(0,1.05)



plt.xticks(np.arange(1,20))



plt.legend(loc=2);
#how to check the number of principal components

sum(pca_hitters.explained_variance_ratio_[:11]) #11 components explaining the 98% of variance
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error,r2_score
pca_lm=LinearRegression()
pca_lm.fit(PC_score.iloc[:,:11],y_train)
pca_lm.coef_
pca_lm.intercept_
pca_lm.score(PC_score.iloc[:,:11],y_train)
Scale_Xtest=scalar.transform(X_test)

Scale_Xtest
pca_test=pca_hitters.transform(Scale_Xtest)

pca_test
pca_test.shape
y_pred_pca=pca_lm.predict(pca_test[:,:11])

y_pred_pca
SSE_pca=np.sum((y_test-y_pred_pca)**2)

SST_pca=np.sum((y_test-np.mean(y_train))**2)

r2_pca=1-SSE_pca/SST_pca

r2_pca #PCA is for reducing columns not for improving performance
rmse_pca=np.sqrt(mean_squared_error(y_test,y_pred_pca))

rmse_pca
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
x_train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
x_test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
x_train.info()
x_test.info()
x_all=pd.concat([x_train,x_test],axis=0)
x_all['Train']=x_all.SalePrice.isnull()
x_all['Train']=x_all.Train.map({False:1,True:0})
x_all.info()
x_all=x_all.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)
cat_col=x_all.drop(['Id','SalePrice','Train'],axis=1).select_dtypes(exclude='number').columns.to_list()
num_col=x_all.drop(['Id','SalePrice','Train'],axis=1).select_dtypes(include='number').columns.to_list()
y=x_all.SalePrice
len(cat_col)
import matplotlib.pyplot as plt
import seaborn as sns
params={'axes.labelsize':20,'xtick.labelsize':12,'ytick.labelsize':12}
plt.rcParams.update(params)

cat_col1=cat_col[:12]
figure,axes=plt.subplots(nrows=int(len(cat_col1)/2)+1,ncols=2,figsize=(20,40))

axes=axes.flatten()

n_delete=(int(len(cat_col1)/2)+1)*2-len(cat_col1)
for i in range(n_delete,0,-1):
    figure.delaxes(axes[-i])

figure.subplots_adjust(wspace=0.2,hspace=0.2)
    
for i in range(len(cat_col1)):
    #if (i%2!=0):
    #    axes[i].yaxis.label.set_visible(False)
    ordering=x_train.groupby(by=cat_col1[i]).SalePrice.median().sort_values().index
    sns.boxplot(y=cat_col1[i],x='SalePrice',data=x_train,ax=axes[i],order=ordering)
    #axes[i].set_xticklabels(labels=x_train[cat_col[i]],rotation=90)
cat_col1=cat_col[12:24]
figure,axes=plt.subplots(nrows=int(len(cat_col1)/2)+1,ncols=2,figsize=(20,40))

axes=axes.flatten()

n_delete=(int(len(cat_col1)/2)+1)*2-len(cat_col1)
for i in range(n_delete,0,-1):
    figure.delaxes(axes[-i])

figure.subplots_adjust(wspace=0.2,hspace=0.2)
    
for i in range(len(cat_col1)):
    #if (i%2!=0):
    #    axes[i].yaxis.label.set_visible(False)
    ordering=x_all[x_all['Train']==1].groupby(by=cat_col1[i]).SalePrice.median().sort_values().index
    sns.boxplot(y=cat_col1[i],x='SalePrice',data=x_all[x_all['Train']==1],ax=axes[i],order=ordering)
    #axes[i].set_xticklabels(labels=x_train[cat_col[i]],rotation=90)
cat_col1=cat_col[24:]
figure,axes=plt.subplots(nrows=int(len(cat_col1)/2)+1,ncols=2,figsize=(20,40))

axes=axes.flatten()

n_delete=(int(len(cat_col1)/2)+1)*2-len(cat_col1)
for i in range(n_delete,0,-1):
    figure.delaxes(axes[-1])

#figure.subplots_adjust(wspace=0.2,hspace=0.2)
    
for i in range(len(cat_col1)):
    #if (i%2!=0):
    #    axes[i].yaxis.label.set_visible(False)
    ordering=x_train.groupby(by=cat_col1[i]).SalePrice.median().sort_values().index
    sns.boxplot(y=cat_col1[i],x='SalePrice',data=x_train,ax=axes[i],order=ordering)
    #axes[i].set_xticklabels(labels=x_train[cat_col[i]],rotation=90)

plt.tight_layout()
len(num_col)
num_col1=num_col[:13]
fig,axes=plt.subplots(nrows=int(len(num_col1)/2)+1,ncols=2,figsize=(20,30),sharey=True)

n_delete=2*(int(len(num_col1)/2)+1)-len(num_col1)

axes=axes.flatten()

for i in range(n_delete,0,-1):
    fig.delaxes(axes[-i])

fig.subplots_adjust(wspace=0.2,hspace=0.2)

for i in range(len(num_col1)):
    axes[i].set_ylabel('SalePrice')
    sns.scatterplot(x=num_col1[i],y='SalePrice',data=x_train,ax=axes[i],hue='SalePrice')
    if (i%2!=0):
        axes[i].yaxis.label.set_visible(False)
    axes[i].grid()
plt.tight_layout()
num_col1=num_col[13:26]
fig,axes=plt.subplots(nrows=int(len(num_col1)/2)+1,ncols=2,figsize=(20,30),sharey=True)

n_delete=2*(int(len(num_col1)/2)+1)-len(num_col1)

axes=axes.flatten()

for i in range(n_delete,0,-1):
    fig.delaxes(axes[-i])

fig.subplots_adjust(wspace=0.2,hspace=0.2)

for i in range(len(num_col1)):
    axes[i].set_ylabel('SalePrice')
    sns.scatterplot(x=num_col1[i],y='SalePrice',data=x_train,ax=axes[i],hue='SalePrice')
    if (i%2!=0):
        axes[i].yaxis.label.set_visible(False)
    axes[i].grid()
plt.tight_layout()
num_col1=num_col[26:]
fig,axes=plt.subplots(nrows=int(len(num_col1)/2)+1,ncols=2,figsize=(20,30),sharey=True)

n_delete=2*(int(len(num_col1)/2)+1)-len(num_col1)

axes=axes.flatten()

for i in range(n_delete,0,-1):
    fig.delaxes(axes[-i])

fig.subplots_adjust(wspace=0.2,hspace=0.2)

for i in range(len(num_col1)):
    axes[i].set_ylabel('SalePrice')
    sns.scatterplot(x=num_col1[i],y='SalePrice',data=x_train,ax=axes[i],hue='SalePrice')
    if (i%2!=0):
        axes[i].yaxis.label.set_visible(False)
    axes[i].grid()
plt.tight_layout()
num_or_cat=['OverallQual','OverallCond','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','Fireplaces','GarageCars','MoSold','YrSold']

fig,axes=plt.subplots(nrows=int(len(num_or_cat)/2)+1,ncols=2,figsize=(20,30))

axes=axes.flatten()

n_delete=2*(int(len(num_or_cat)/2)+1)-len(num_or_cat)
for i in range(n_delete,0,-1):
    fig.delaxes(axes[-i])

for i in range(len(num_or_cat)): 
    sns.countplot(num_or_cat[i],data=x_all,ax=axes[i],alpha=0.5,hue=num_or_cat[i])
    sns.countplot(num_or_cat[i],data=x_train,ax=axes[i],hue=num_or_cat[i])
    axes[i].legend_.remove()
    axes[i].grid()

plt.tight_layout()
x_all['MoSold']=x_all['MoSold'].astype('object')
x_all['YrSold']=x_all['YrSold'].astype('object')
cat_col.extend(['MoSold','YrSold'])
num_col.remove('MoSold')
num_col.remove('YrSold')
num_col1=num_col[:12]
fig,axes=plt.subplots(nrows=int(len(num_col1)/2)+1,ncols=2,figsize=(20,30))
fig.subplots_adjust(wspace=0.2,hspace=0.2)

axes=axes.flatten()

n_delete=2*int(len(num_col1)/2+1)-len(num_col1)
for i in range(n_delete,0,-1):
    fig.delaxes(axes[-i])
    
for i in range(len(num_col1)):
    axes[i].hist(x_all[num_col1[i]],bins=75)
    axes[i].set_title(num_col1[i])
    axes[i].grid()
num_col1=num_col[12:24]
fig,axes=plt.subplots(nrows=int(len(num_col1)/2)+1,ncols=2,figsize=(20,30))
fig.subplots_adjust(wspace=0.2,hspace=0.2)

axes=axes.flatten()

n_delete=2*int(len(num_col1)/2+1)-len(num_col1)
for i in range(n_delete,0,-1):
    fig.delaxes(axes[-i])
    
for i in range(len(num_col1)):
    axes[i].hist(x_all[num_col1[i]],bins=75)
    axes[i].set_title(num_col1[i])
    axes[i].grid()
num_col1=num_col[24:]
fig,axes=plt.subplots(nrows=int(len(num_col1)/2)+1,ncols=2,figsize=(20,30))
fig.subplots_adjust(wspace=0.2,hspace=0.2)

axes=axes.flatten()

n_delete=2*int(len(num_col1)/2+1)-len(num_col1)
for i in range(n_delete,0,-1):
    fig.delaxes(axes[-i])
    
for i in range(len(num_col1)):
    axes[i].hist(x_all[num_col1[i]],bins=75)
    axes[i].set_title(num_col1[i])
    axes[i].grid()
time_ind=['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold']
x_all['garage_life']=x_all['YrSold'].astype('int64')-x_all['GarageYrBlt']
x_all['house_life']=x_all['YrSold'].astype('int64')-x_all['YearBuilt']
x_all.drop(['YearBuilt','YearRemodAdd','GarageYrBlt'],axis=1,inplace=True)
x_all['YrSold']=x_all['YrSold'].astype('object')
x_all=x_all.reset_index()
x_all.describe()
cat_col=x_all.drop(['Id','Train','SalePrice'],axis=1).select_dtypes(exclude='number').columns.to_list()
num_col=x_all.drop(['Id','Train','SalePrice'],axis=1).select_dtypes(include='number').columns.to_list()
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
num_imputer=IterativeImputer()
col_imputer=SimpleImputer(strategy='most_frequent')

x_all.loc[x_all['Train']==1,num_col]=num_imputer.fit_transform(x_all[x_all['Train']==1][num_col].values)
x_all.loc[x_all['Train']==0,num_col]=num_imputer.transform(x_all[x_all['Train']==0][num_col].values)
x_all.loc[x_all['Train']==1,cat_col]=col_imputer.fit_transform(x_all[x_all['Train']==1][cat_col].values)
x_all.loc[x_all['Train']==0,cat_col]=col_imputer.transform(x_all[x_all['Train']==0][cat_col].values)
x_all.info()
x_corr=abs(x_all.drop(['Id','Train','SalePrice'],axis=1)[num_col].corr())
fig,axes=plt.subplots(figsize=(10,6))
sns.heatmap(x_corr,ax=axes)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
print(x_corr)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
skewed=x_all[num_col].skew()[abs(x_all[num_col].skew())>=4].index.to_list()
ptransformer=PowerTransformer(standardize=False)
x_all.loc[x_all['Train']==1,skewed]=ptransformer.fit_transform(x_all.loc[x_all['Train']==1,skewed])
x_all.loc[x_all['Train']==0,skewed]=ptransformer.transform(x_all.loc[x_all['Train']==0,skewed])
num_processor=ColumnTransformer(transformers=[('scaler',StandardScaler(),num_col)])
x_all.loc[x_all['Train']==1,num_col]=num_processor.fit_transform(x_all.loc[x_all['Train']==1,num_col])
x_all.loc[x_all['Train']==0,num_col]=num_processor.transform(x_all.loc[x_all['Train']==0,num_col])

cat_processor=ColumnTransformer(transformers=[('encoder',OneHotEncoder(sparse=False,handle_unknown='ignore'),cat_col)])
x_cat=pd.DataFrame(cat_processor.fit_transform(x_all.drop(['Id','SalePrice'],axis=1)))
x_all=pd.concat([x_all[num_col+['Train','SalePrice']],x_cat],axis=1)
x_train=x_all.loc[x_all['Train']==1,:].drop(['Train','SalePrice'],axis=1)
x_test=x_all.loc[x_all['Train']==0,:].drop(['Train','SalePrice'],axis=1)
y=x_all.loc[x_all['Train']==1,'SalePrice']
x_train.info()
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
from sklearn.linear_model import LinearRegression
Lasso_score=cross_val_score(Lasso(),x_train,y,cv=5,n_jobs=-1,error_score='neg_root_mean_squared_error')
Ridge_score=cross_val_score(Ridge(),x_train,y,cv=5,n_jobs=-1,error_score='neg_root_mean_squared_error')
rf_score=cross_val_score(RandomForestRegressor(n_estimators=100),x_train,y,cv=5,n_jobs=-1,error_score='neg_root_mean_squared_error')
gbrt_score=cross_val_score(XGBRegressor(n_estimators=100),x_train,y,cv=5,n_jobs=-1,error_score='neg_root_mean_squared_error')
stacking_score=cross_val_score(StackingCVRegressor(regressors=[Lasso(),Ridge(),RandomForestRegressor(n_estimators=100),XGBRegressor(n_estimators=100)],meta_regressor=LinearRegression(n_jobs=-1)),x_train,y,cv=5,n_jobs=-1,error_score='neg_root_mean_squared_error')
scores={'Lasso':Lasso_score,'Ridge':Ridge_score,'RandomForestRegression':rf_score,'GBRT':gbrt_score,'stacking':stacking_score}
scores=pd.DataFrame(scores)
models=scores.columns.to_list()
score=[Lasso_score,Ridge_score,rf_score,gbrt_score,stacking_score]

fig,ax=plt.subplots()
sns.boxplot(x=models,y=score,ax=ax)
plt.xticks(rotation=90)
from sklearn.model_selection import train_test_split
x_1,x_2,y_1,y_2=train_test_split(x_train,y,test_size=0.1)
stacking=StackingCVRegressor(regressors=[Lasso(),Ridge(),RandomForestRegressor(n_estimators=200),XGBRegressor(n_estimators=200)],meta_regressor=LinearRegression(),n_jobs=-1)
stacking.fit(x_1,y_1)
y_2_predict=stacking.predict(x_2)

fig,ax=plt.subplots()
sns.scatterplot(x=y_2,y=y_2_predict,ax=ax)
sns.lineplot(x=y_2,y=y_2)
from sklearn.model_selection import GridSearchCV
params={'alpha':[0.001,0.01,0.1,1,10,100]}
grid=GridSearchCV(Lasso(max_iter=10000),param_grid=params,cv=3)
grid.fit(x_train,y)
grid.best_params_
lasso=Lasso(alpha=100)
lasso.fit(x_1,y_1)
y_2_predict=lasso.predict(x_2)

fig,ax=plt.subplots()
sns.scatterplot(x=y_2,y=y_2_predict,ax=ax)
sns.lineplot(x=y_2,y=y_2)

print(cross_val_score(lasso,x_train,y))
params={'n_estimators':[100,250,500,750,1000],'max_depth':[2,3,4,5,6,7,8],'n_jobs':[-1]}
grid=GridSearchCV(RandomForestRegressor(),param_grid=params,cv=5)
grid.fit(x_train,y)
grid.best_params_
rf=RandomForestRegressor(max_depth=8,n_estimators=100,n_jobs=-1)
rf.fit(x_1,y_1)
y_2_predict=rf.predict(x_2)

fig,ax=plt.subplots()
sns.scatterplot(x=y_2,y=y_2_predict,ax=ax)
sns.lineplot(x=y_2,y=y_2)

print(cross_val_score(rf,x_train,y))
params={'n_estimators':[100,200,300,400,500,600,800,1000],'max_depth':[3,4,5,6,7,8,9,10],'eta':[0.01,0.05,0.1,0.3,0.5,0.8,1],'reg_alpha':[0.01,0.1,1,10],'n_jobs':[-1]}
grid=GridSearchCV(XGBRegressor(),param_grid=params)
grid.fit(x_train,y)
grid.best_params_

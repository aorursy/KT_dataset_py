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
#for basic analysis

import pandas as pd

import numpy as np



#for visualisations

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



#for stats & modelling

import scipy.stats as stats

from sklearn import metrics

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor
#read the data into pandas dataframe

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
train.head()
train.info()
# UDF for descriptive analysis of numeric vars.

def con_stats(c):

    return pd.Series({'Count':c.count(),'NaNs':c.isnull().sum(),'%NaNs':c.isnull().sum()*100/c.shape[0],'Sum':c.sum(),'Mean':c.mean(),'Std.Dev':c.std(),

                      'Coef.of.Var':c.std()/c.mean(),'Min':c.min(),'P1':c.quantile(.01),'P5':c.quantile(.05),'P10':c.quantile(.1),'P25':c.quantile(.25),

                      'P50':c.quantile(.5),'P90':c.quantile(.9),'P95':c.quantile(.95),'P99':c.quantile(.99),'Max':c.max()})
# UDF for descriptive analysis for categorical vars

def cat_stats(v):

    return pd.Series({'Count':v.count(),'NaNs':v.isnull().sum(),'%NaNs':v.isnull().sum()*100/v.shape[0],'#.Uniques':v.unique().shape[0],

                     'Mode':v.mode()[0]})
# UDF for Missing Value treatment 

def miss(m):

    if (m.dtype=='int64')|(m.dtype=='float64'):

        m.fillna(m.mean(),inplace=True)

    elif m.dtype=='object':

        m.fillna(m.mode()[0],inplace=True)  #m.mode()[0] means, fill with first occurence of mode incase of multimodal

    return m
# UDF for outlier treatment

def outs(o):

    if (o.dtype=='int64')|(o.dtype=='float64'):

        o.clip(lower=o.quantile(0.01), upper=o.quantile(0.95), inplace=True)

        return o

    else:

        return o
# UDF for dummy var creation

def dums(d):

    d=pd.get_dummies(d,drop_first=True, prefix='Dum')

    return d
#don't need ID column for analysis

train.drop(columns=['Id'],inplace=True)
train.select_dtypes(exclude='object').apply(con_stats).T.round(2)
train.drop(columns = ['YrSold','GarageYrBlt','YearBuilt','YearRemodAdd'], inplace=True)
train.select_dtypes(include='object').apply(cat_stats).T
x=train[['Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',

'PoolQC','Fence','MiscFeature']].copy()



train.drop(columns=['Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',

'PoolQC','Fence','MiscFeature'],inplace=True)



x.fillna('None',inplace=True)



train=pd.concat([train,x],axis=1)
train.isnull().any().any()
#train.GarageYrBlt.fillna(train.GarageYrBlt.mode()[0],inplace=True)  #don't want to fill year with mean value of year

train=train.apply(miss)
train.isnull().any().any()
# Outlier Treatment 

sns.set(style='whitegrid')

f,ax=plt.subplots(1,3,figsize=(16,5))

sns.boxplot(train.SalePrice,ax=ax[0],color='purple', whis=[1,99])

sns.boxplot(train.LotArea,ax=ax[1],color='red', whis=[1,99])

sns.boxplot(train.GrLivArea,ax=ax[2],color='green', whis=[1,99])

plt.suptitle('Plots showing Outliers')
# treating outliers, by clipping them to 1st and 95th percentiles

train=train.apply(outs)
f,ax=plt.subplots(1,3,figsize=(16,5))

sns.boxplot(train.SalePrice,ax=ax[0],color='purple', whis=[1,99])

sns.boxplot(train.LotArea,ax=ax[1],color='red', whis=[1,99])

sns.boxplot(train.GrLivArea,ax=ax[2],color='green', whis=[1,99])

plt.suptitle('Post outlier treatment')
mapper = {'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5,

          'No':1, 'Mn':2, 'Av':3,

          'Unf':1,'LwQ':2, 'Rec':3, 'BLQ':4,'ALQ':5, 'GLQ':6,

          'IR3':0, 'IR2':1, 'IR1':2, 'Reg':3,

          'ELO':0, 'NoSeWa':1, 'NoSewr':2, 'AllPub':3,

          'Low':0,'HLS':1,'Bnk':2, 'Lvl':3,

          'Sev':0, 'Mod':1, 'Gtl':2,

          'MnWw':1,'GdWo':2, 'MnPrv':3,'GdPrv':4,

          'RFn':2,'Fin':3,

          'Pave':1,'Grvl':2,

          'N':0,'P':1,'Y':2}



def ranker(c):

    c=c.map(mapper)

    return c
ranked_vars = train[['ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','HeatingQC','KitchenQual','FireplaceQu','GarageQual',

 'GarageCond','PoolQC','LotShape', 'Utilities','LandContour','LandSlope','Fence','GarageFinish','Street','Alley','CentralAir','PavedDrive']].apply(ranker)



ranked_vars.head()
train.drop(columns=['ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','HeatingQC','KitchenQual','FireplaceQu','GarageQual',

 'GarageCond','PoolQC','LotShape', 'Utilities','LandContour','LandSlope','Fence','GarageFinish','Street','Alley','CentralAir','PavedDrive'],inplace=True)  # dropping original columns
train=pd.concat([train,ranked_vars],axis=1) #concatenating new ordinal vars
train=dums(train)
train.info()
# Normality of target

sns.set(style='whitegrid')

sns.distplot(train.SalePrice, fit=stats.norm,color='red')

print('Skewness :',stats.skew(train.SalePrice))   # still skewed even after outlier treatment
#applying log transform on target

train['ln_SalePrice']=np.log1p(train.SalePrice)

sns.distplot(train.ln_SalePrice, fit=stats.norm, color='blue')

print('Skewness :',stats.skew(train.ln_SalePrice))
# Linearity with target

f1,ax=plt.subplots(2,2,figsize=(14,8))

p1=(stats.pearsonr(train.OverallQual, train.ln_SalePrice)[0])

sns.scatterplot(y='OverallQual',x='ln_SalePrice',data=train,ax=ax[0,0],label=['pearson_r-', p1.round(2)],color='blue')

p2=(stats.pearsonr(train.TotalBsmtSF, train.ln_SalePrice)[0])

sns.scatterplot(y='TotalBsmtSF',x='ln_SalePrice',data=train,ax=ax[0,1],label=['pearson_r-', p2.round(2)],color='green')

p3=(stats.pearsonr(train.GarageArea, train.ln_SalePrice)[0])

sns.scatterplot(y='GarageArea',x='ln_SalePrice',data=train,ax=ax[1,0],label=['pearson_r-', p3.round(2)],color='red')

p4=(stats.pearsonr(train.GrLivArea, train.ln_SalePrice)[0])

sns.scatterplot(y='GrLivArea',x='ln_SalePrice',data=train,ax=ax[1,1],label=['pearson_r-', p4.round(2)],color='purple')

plt.suptitle('Linearity with Target')
f=[]



for i in train.select_dtypes(exclude='uint8').columns:

    f.append(stats.pearsonr(train[i],train.ln_SalePrice)[0].round(2))



pr_df=pd.DataFrame({'Features':train.select_dtypes(exclude='uint8').columns, 'Pearson_r':f})

pr_df.sort_values(by='Pearson_r',ascending=False)[0:18]  # top 18 which shows very high linearity with target
train.drop(columns=['LowQualFinSF','KitchenAbvGr','3SsnPorch','PoolArea','MiscVal'],inplace=True) 

#removing these as they filled with only zeros and are like redundant variables (POOL AREA taken care by POOLQC)
# Multicollinearity

cormat=train.select_dtypes(exclude='uint8').corr()

plt.figure(figsize=(13,9))

sns.heatmap(cormat,linewidths=.5, cmap='RdYlGn')
mm=MinMaxScaler()

scaled=mm.fit_transform(train)
scaled
# create dataframe of the scaled features

train=pd.DataFrame(scaled, columns=train.columns)

train.head()
# we'll work with ln_SalePrice now onwards

original_SalePrice=train.SalePrice.copy()

train.drop(columns=['SalePrice'], inplace=True)
train.shape
# separating feature variables & target

X = train[train.columns.difference(['ln_SalePrice'])]

y = train['ln_SalePrice']
# train test split, this splits the train data in train(70):test(30) ratio

train_X,test_X,train_y,test_y = train_test_split(X,y, test_size=0.3, random_state=12345)
lr=LinearRegression()

lr.fit(train_X,train_y)
#prediction for train and test

lr_pred = lr.predict(train_X)

lr_test_pred = lr.predict(test_X)
# evaluation metrics calculation

train_lr_mse = metrics.mean_squared_error(train_y,lr_pred).round(3)

train_lr_mae = metrics.mean_absolute_error(train_y,lr_pred).round(3)

train_lr_r2 = metrics.r2_score(train_y,lr_pred).round(3)



test_lr_mae = metrics.mean_absolute_error(test_y,lr_test_pred).round(3)

test_lr_mse = metrics.mean_squared_error(test_y,lr_test_pred).round(3)

test_lr_r2 = metrics.r2_score(test_y,lr_test_pred).round(3)
lin_reg_eva= pd.DataFrame(index=['R2','MAE','MSE'],columns=['Train','Test'],data=[[train_lr_r2,test_lr_r2],[train_lr_mae,test_lr_mae],

                                                                                 [train_lr_mse,test_lr_mse]])

lin_reg_eva
rs=12345

rr=Ridge(random_state=rs)

para2 = [{'alpha':[0.0,.001,.01,.1,10,100]}]

grid_rr = GridSearchCV(rr, param_grid = para2, n_jobs=-1, cv=5)    # grid search runs model using different combinations of parameters
grid_rr.fit(train_X, train_y)
# to get the best parameters

grid_rr.best_params_
# using the best parameter in our model

rr=Ridge(alpha=10, random_state=rs)

rr.fit(train_X,train_y)

rr_train_pred = rr.predict(train_X)

rr_test_pred = rr.predict(test_X)
# evaluation metrics calculation

train_rr_r2 = (metrics.r2_score(train_y, rr_train_pred)).round(3)

test_rr_r2 = (metrics.r2_score(test_y, rr_test_pred)).round(3)



train_rr_mse = (metrics.mean_squared_error(train_y, rr_train_pred)).round(3)

test_rr_mse = (metrics.mean_squared_error(test_y, rr_test_pred)).round(3)



train_rr_mae = (metrics.mean_absolute_error(train_y, rr_train_pred)).round(3)

test_rr_mae = (metrics.mean_absolute_error(test_y, rr_test_pred)).round(3)
rs = 12345

dt=DecisionTreeRegressor(random_state = rs)



para = [ {'max_depth':[6,7,8,9,10],

          'max_features':[90,100,110,120,130]}]



grid_tree = GridSearchCV(dt, param_grid=para, n_jobs=-1, cv=5)
grid_tree.fit(train_X, train_y)
grid_tree.best_params_
# using the best parameter in our model

dtr = DecisionTreeRegressor(max_depth =6, max_features =120,random_state=rs)

dtr.fit(train_X,train_y)

dtr_train_pred = dtr.predict(train_X)

dtr_test_pred = dtr.predict(test_X)
# evaluation metrics

train_dtr_r2 = (metrics.r2_score(train_y,dtr_train_pred)).round(3)

test_dtr_r2 = (metrics.r2_score(test_y,dtr_test_pred)).round(3)



train_dtr_mse = (metrics.mean_squared_error(train_y,dtr_train_pred)).round(3)

test_dtr_mse = (metrics.mean_squared_error(test_y,dtr_test_pred)).round(3)



train_dtr_mae = (metrics.mean_absolute_error(train_y,dtr_train_pred)).round(3)

test_dtr_mae = (metrics.mean_absolute_error(test_y,dtr_test_pred)).round(3)
rf=RandomForestRegressor(random_state = rs)

para2 = [{'max_features':[50,60,70,80],

          'n_estimators':[60,70,80,90]}]

grid_rf = GridSearchCV(rf, param_grid = para2, cv=5, n_jobs=-1)
grid_rf.fit(train_X,train_y)
grid_rf.best_params_
# using the best parameter in our model

rfm = RandomForestRegressor(max_features = 60, n_estimators = 60, random_state=rs)

rfm.fit(train_X,train_y)

rfm_train_pred = rfm.predict(train_X)

rfm_test_pred = rfm.predict(test_X)
# evaluation metrics

train_rfm_r2 = (metrics.r2_score(train_y, rfm_train_pred)).round(3)

test_rfm_r2 = (metrics.r2_score(test_y, rfm_test_pred)).round(3)



train_rfm_mse = (metrics.mean_squared_error(train_y, rfm_train_pred)).round(3)

test_rfm_mse = (metrics.mean_squared_error(test_y, rfm_test_pred)).round(3)



train_rfm_mae = (metrics.mean_absolute_error(train_y, rfm_train_pred)).round(3)

test_rfm_mae = (metrics.mean_absolute_error(test_y, rfm_test_pred)).round(3)
# random forests imp features

imp_fet = pd.DataFrame({'Features':train_X.columns, 'Weightage':rfm.feature_importances_})

imp_fet.sort_values(by='Weightage',ascending=False).head()
f,ax=plt.subplots(1,3,figsize=(19,7))

sns.lineplot(x=['Ridge','D.tree','RandomForest'], y=[train_rr_r2, train_dtr_r2, train_rfm_r2], color='blue',ax=ax[0], label='Train')

sns.lineplot(x=['Ridge','D.tree','RandomForest'], y=[test_rr_r2, test_dtr_r2, test_rfm_r2], color='red',ax=ax[0], label='Test')

ax[0].set_title('R2 Comparison for Train & Test')



sns.lineplot(x=['Ridge','D.tree','RandomForest'], y=[train_rr_mae, train_dtr_mae, train_rfm_mae], color='blue',ax=ax[1],label='Train')

sns.lineplot(x=['Ridge','D.tree','RandomForest'], y=[test_rr_mae, test_dtr_mae, test_rfm_mae], color='red',ax=ax[1],label='Test')

ax[1].set_title('MAE Comparison for Train & Test')



sns.lineplot(x=['Ridge','D.tree','RandomForest'], y=[train_rr_mse, train_dtr_mse, train_rfm_mse], color='blue',ax=ax[2],label='Train')

sns.lineplot(x=['Ridge','D.tree','RandomForest'], y=[test_rr_mse, test_dtr_mse, test_rfm_mse], color='red',ax=ax[2],label='Test')

ax[2].set_title('MSE Comparison for Train & Test')



plt.suptitle("Model Evaluation Metrics Comparison")
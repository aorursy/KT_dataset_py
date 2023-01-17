import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

pd.pandas.set_option('display.max_columns', None)
df=pd.read_csv('train2 (2).csv')

df.head()
df.isnull().sum()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(df,df['SalePrice'],test_size=0.1,random_state=0)
nanpercent=[x for x in df.columns if df[x].isnull().sum()>1 and df[x].dtypes=='O']

for x in nanpercent:

    print("{}: {}% missing values".format(x,np.round(df[x].isnull().mean(),4)))
def replacenan(df,nanpercent):

    df1=df.copy()

    df1[nanpercent]=df1[nanpercent].fillna('missing')

    return df1

df=replacenan(df,nanpercent)

df[nanpercent].isnull().sum()
df.head()
numericalnan=[x for x in df.columns if df[x].isnull().sum()>1 and df[x].dtypes!='O']

for x in numericalnan:

    print("{} : {}% missing values".format(x,np.round(df[x].isnull().mean(),4)))
for x in numericalnan:

    med=df[x].median()

    df[x+'nan']=np.where(df[x].isnull(),1,0)

    df[x].fillna(med,inplace=True)

df[numericalnan].isnull().sum()
df.head(50)
for x in ['YearBuilt','YearRemodAdd','GarageYrBlt']:

       

    df[x]=df['YrSold']-df[x]
df.head()
df[['YearBuilt','YearRemodAdd','GarageYrBlt']].head()
import numpy as np

num_features=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']



for x in num_features:

    df[x]=np.log(df[x])
df.head()
catfeatures=[x for x in df.columns if df[x].dtypes=='O' ]

catfeatures
for x in catfeatures:

    temp=df.groupby(x)['SalePrice'].count()/len(df)

    temp_df=temp[temp>0.01].index

    df[x]=np.where(df[x].isin(temp_df),df[x],'Rarevalue')
df.head(100)
for x in catfeatures:

    labels_ordered=df.groupby([x])['SalePrice'].mean().sort_values().index

    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}

    df[x]=df[x].map(labels_ordered)
df.head()
scalefeature=[x for x in df.columns if x not in ['Id','Saleprice']]
scalefeature
feature_scale=[x for x in df.columns if x not in ['Id','SalePrice']]



from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

scaler.fit(df[feature_scale])
scaler.transform(df[feature_scale])
df1 = pd.concat([df[['Id', 'SalePrice']].reset_index(drop=True),

                    pd.DataFrame(scaler.transform(df[feature_scale]), columns=feature_scale)],

                    axis=1)
df1.shape

df1.to_csv('X_train.csv',index=False)
df=pd.read_csv('X_train.csv')
df.head()
pd.pandas.set_option('display.max_columns',None)
y_train= df[['SalePrice']]
x_train= df.drop(['Id','SalePrice'],axis=1)
from sklearn.linear_model import Lasso

from sklearn.feature_selection import SelectFromModel
selmodel=SelectFromModel(Lasso(alpha=0.005,random_state=0))

selmodel.fit(x_train,y_train)
selmodel.get_support()
selected=x_train.columns[(selmodel.get_support())]
selected
X_test[selected].head()
x_train=x_train[selected]
x_train.head()
params={'colsample_bytree':[0.4,0.6,0.8],

       'gamma':[0,0.03,0.1,0.3],

       'min_child_weight':[1.5,6,10],

       'learning_rate':[0.1,0.07],

       'max_depth':[3,5],

       'n_estimastors':[10000],

       'reg_alpha':[1e-5, 1e-2,0.45],

       'subsample':[0.6,0.95]}

from sklearn.model_selection import GridSearchCV

import xgboost
xgb_model=xgboost.XGBRegressor(learningrate=0.1,n_estimator=1000,max_depth=5,min_child_weight=1,gamma=0,subsample=0.8,colsample_bytree=0.8,nthread=6,scale_pos_eight=1,seed=27)
gsearch1=GridSearchCV(estimator= xgb_model,param_grid=params,n_jobs=6,iid=False,verbose=10,scoring='neg_mean_squared_error')
def timer(start_time=None):

    if not start_time:

        start_time = datetime.now()

        return start_time

    elif start_time:

        thour,temp_sec=divmod((datetime.now() - start_time).total_seconds(),3600)

        tmin, tsec = divmod(temp_sec,60)

        print('\n Time taken:%i hours %i minutes %s seconds'%(thour.tmin,round(tsec,2)))
gsearch1.best_params_
classi = xgboost.XGBRegressor(colsample_bytree= 0.4,

 gamma= 0,

 learning_rate= 0.1,

 max_depth= 5,

 min_child_weight= 1.5,

 n_estimastors= 10000,

 reg_alpha= 0.01,

 subsample= 0.95)
from sklearn.model_selection import cross_val_score

score=cross_val_score(classi , x_train,y_train,cv=10)
score
score.mean()
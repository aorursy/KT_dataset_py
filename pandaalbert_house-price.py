import pandas as pd 

import numpy as np 

import matplotlib as mpl 

import matplotlib.pyplot as plt

from sklearn.preprocessing import power_transform,StandardScaler,OneHotEncoder

from sklearn.impute import SimpleImputer,KNNImputer

from sklearn.decomposition import PCA

from sklearn.svm import SVR

from sklearn.cross_decomposition import PLSRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.multiclass import OneVsRestClassifier

from sklearn.model_selection import cross_validate,GridSearchCV

from sklearn.metrics import accuracy_score,mean_squared_error

from lightgbm.sklearn import LGBMRegressor

from sklearn.linear_model import Ridge

from sklearn.ensemble import StackingRegressor,RandomForestRegressor

from scipy.stats import boxcox_normmax,skew

from scipy.special import boxcox1p

from statsmodels.formula.api import ols

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")
train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.shape
test.shape
train_y=train['SalePrice']

train_y=np.log1p(train_y)

train=train.drop('SalePrice',axis=1)

df=pd.concat([train,test],axis=0)
df.shape
for col in ['Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu',

            'GarageType','GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature']:

    df[col]=df[col].fillna(value='No')
tmp=df.astype(object)

deg_=[]

for col in list(tmp.columns):

    tol=tmp[col].value_counts()

    if (tol.iloc[0]//tol.iloc[1]>30) & (sum(tol.iloc[1:])/tmp.shape[0]<=0.1):

        print('degenerate column:',col,': with ratio',sum(tol.iloc[1:])/tmp.shape[0],'and',

             tol.iloc[0]//tol.iloc[1])

        deg_.append(col)
df=df.drop(deg_,axis=1)
pd.DataFrame(df.isnull().sum()/df.shape[0]).sort_values(0,ascending=False)[:20]
num_=[]

for col in list(df.columns)[1:]:

    if np.dtype(train[col])!=object:

        num_.append(col)



df_num=df[num_]

df_obj=df.drop(num_,axis=1)
tmp=df_num.copy()

tmp['y']=train_y

tmp=tmp.corr()
high=list(pd.DataFrame(tmp['y']).sort_values('y',ascending=False).index[1:11])
plt.figure(figsize=(20,8))

i=1

for col in high:

    plt.subplot(2,5,i)

    i+=1

    sns.regplot(x=df[col][:1460],y=train_y,scatter_kws={'s':5})
df_num.describe().T
df_num.hist(figsize=(20,20))
df_num.loc[df_num.GarageYrBlt>2020,'GarageYrBlt']=np.nan



# for col in list(df_num.columns):

#     if abs(skew(df_num[col]))>0.5:

#         df_num.loc[df_num[col]>df_num[col].quantile(0.75),col]=np.nan

#         df_num.loc[df_num[col]<df_num[col].quantile(0.25),col]=np.nan
df_num.hist(figsize=(20,20))
sns.regplot(x=df_num['OverallQual'][:1460],y=train_y,scatter_kws={'s':5})

plt.hlines(y=12.3,xmin=8,xmax=10)
df_num['y']=train_y
df_num.loc[(df_num.OverallQual==4)&(df_num.y>12.4),'OverallQual']=np.nan

df_num.loc[(df_num.OverallQual==4)&(df_num.y<10.7),'OverallQual']=np.nan

df_num.loc[(df_num.OverallQual==3)&(df_num.y<10.7),'OverallQual']=np.nan

df_num.loc[(df_num.OverallQual==2)&(df_num.y<10.7),'OverallQual']=np.nan

df_num.loc[(df_num.OverallQual==10)&(df_num.y<12.3),'OverallQual']=np.nan
sns.regplot(x=df_num['OverallQual'][:1460],y=train_y,scatter_kws={'s':5})

plt.hlines(y=12.3,xmin=8,xmax=10)
sns.regplot(x=df_num['GrLivArea'][:1460],y=train_y,scatter_kws={'s':5})

# plt.hlines(y=12.75,xmin=1350,xmax=1700)

plt.hlines(y=11,xmin=1100,xmax=1400)
df_num.loc[(df_num.y<11),'GrLivArea']=np.nan

# df_num.loc[(df_num.y>12.75),'GrLivArea']=np.nan
sns.regplot(x=df_num['GrLivArea'][:1460],y=train_y,scatter_kws={'s':5})

# plt.hlines(y=12.75,xmin=1350,xmax=1700)

plt.hlines(y=11,xmin=1100,xmax=1400)
sns.regplot(x=df_num['GarageCars'][:1460],y=train_y,scatter_kws={'s':5})

plt.hlines(y=10.7,xmin=0,xmax=1.5)
df_num.loc[(df_num.y<10.7),'GarageCars']=np.nan
sns.regplot(x=df_num['GarageArea'][:1460],y=train_y,scatter_kws={'s':5})

plt.hlines(y=10.7,xmin=0,xmax=400)

plt.hlines(y=12.6,xmin=1200,xmax=1500)
df_num.loc[(df_num.y<10.7),'GarageArea']=np.nan

df_num.loc[(df_num.y<12.6)&(df_num.GarageArea>1200),'GarageArea']=np.nan
plt.figure(figsize=(20,8))

i=1

for col in high:

    plt.subplot(2,5,i)

    i+=1

    sns.regplot(x=df_num[col][:1460],y=train_y,scatter_kws={'s':5})
# imp=SimpleImputer(strategy='mean')

# df_num=pd.DataFrame(imp.fit_transform(df_num),columns=df_num.columns)



knn=KNNImputer(n_neighbors=8)

df_num=pd.DataFrame(knn.fit_transform(df_num),columns=df_num.columns)
df_num=df_num.drop('y',axis=1)
for col in list(df_num.columns):

    if abs(skew(df_num[col]))>0.5:

        df_num[col]=boxcox1p(df_num[col],boxcox_normmax(df_num[col]+1))
df_num.hist(figsize=(30,15))

plt.yticks(ticks=[])

plt.xticks(ticks=[])
std=StandardScaler()

df_num=pd.DataFrame(std.fit_transform(df_num),columns=df_num.columns)
pca=PCA(n_components=25)

df_num=pd.DataFrame(pca.fit_transform(df_num),columns=['PC'+str(i) for i in range(1,26)])
df_num
plt.figure(figsize=(25,18),dpi=100)

for i in range(1,21):

    plt.subplot(5,4,i)

    sns.regplot(x=df_num.iloc[:1460,(i-1)],y=train_y,scatter_kws={'s':1,'alpha':0.6}) 

    plt.yticks(ticks=[])

    plt.xticks(ticks=[])
df_num['y']=train_y

df_num.loc[(df_num.PC1>7.5)&(df_num.y<12.5),'PC1']=np.nan

df_num=df_num.drop('y',axis=1)
imp=SimpleImputer(strategy='median')

df_num=pd.DataFrame(imp.fit_transform(df_num),columns=df_num.columns)
sns.regplot(x=df_num.iloc[:1460,0],y=train_y,scatter_kws={'s':2,'alpha':0.6},lowess=True)
df_num
df_obj=df_obj.drop('Id',axis=1)

pd.DataFrame(df_obj.isnull().sum()).sort_values(0,ascending=False)[:10]
median_col=pd.DataFrame(df_obj.isnull().sum()).sort_values(0,ascending=False).index[1:7]

median_col
for col in list(median_col):

    df_obj[col]=df_obj[col].fillna(df_obj[col].value_counts().index[0])
df_obj_MasVnrType=df_obj['MasVnrType']

df_obj_without=df_obj.drop('MasVnrType',axis=1)
df_obj_nona=df_obj.dropna(axis=0)

df_obj_MasVnrType_na=df_obj_nona['MasVnrType']

df_obj_without_na=df_obj_nona.drop('MasVnrType',axis=1)
onehot=OneHotEncoder()

df_obj_without_na=onehot.fit_transform(df_obj_without_na)
# for n in [2,4,8,16,32]:

#     knn_or=OneVsRestClassifier(KNeighborsClassifier(weights='distance',n_neighbors=n))

#     knn_or.fit(X=df_obj_without_na,y=df_obj_MasVnrType_na)

#     print(n,accuracy_score(df_obj_MasVnrType_na,knn_or.predict(df_obj_without_na)))
knn_or=OneVsRestClassifier(KNeighborsClassifier(weights='distance',n_neighbors=8))

knn_or.fit(X=df_obj_without_na,y=df_obj_MasVnrType_na)

Mas_pred=knn_or.predict(onehot.fit_transform(df_obj_without))
df_obj.loc[df_obj.MasVnrType.isnull()==True,'MasVnrType']=Mas_pred[df_obj.MasVnrType.isnull()]
df_obj['y']=train_y
one_hot_=[]

for col in list(df_obj.columns)[:-1]:

    tmp=df_obj[[col,'y']]

    tmp=tmp.groupby(col).mean().reset_index()

    if (max(tmp.y)-min(tmp.y))<=(2*np.std(train_y)):

        one_hot_.append(col)

len(one_hot_)
df_obj_one=df_obj[one_hot_]

df_obj_ordinal=df_obj.drop(one_hot_,axis=1)
df_obj_one=onehot.fit_transform(df_obj_one)
for col in list(df_obj_ordinal.columns)[:-1]:

    tmp=df_obj_ordinal[[col,'y']].groupby(col).mean().reset_index()

    tmp['y']=tmp['y']/sum(tmp['y'])

    change={}

    for i,j in zip(tmp[col],tmp['y']):

        change[i]=j

    df_obj_ordinal[col]=df_obj_ordinal[col].replace(change)
df_obj_ordinal=df_obj_ordinal.drop('y',axis=1)
df_obj=np.concatenate((df_obj_ordinal,df_obj_one.toarray()),axis=1)
df_num=df_num.values
df_all=np.concatenate((df_num,df_obj),axis=1)
df_train=df_all[:1460,:]

df_test=df_all[1460:,:]
rid=Ridge(alpha=10)

rid.fit(df_train,train_y)

y_pred=rid.predict(df_train)
data=pd.DataFrame({'y_pred':y_pred,'y_actual':train_y})
regression=ols('y_actual~y_pred',data=data).fit()

test=regression.outlier_test()
test.sort_values('unadj_p')[:10]
remain_index=test.loc[test['bonf(p)']>=0.9].index
df_train=df_train[list(remain_index)]
train_y=train_y[list(remain_index)]
pls=PLSRegression()

best_clf = GridSearchCV(pls,scoring='neg_root_mean_squared_error',cv=10,n_jobs=1,

                        param_grid={'n_components':[2,4,8,16]})

best_clf.fit(df_train,train_y)
best_clf.best_score_
best_clf.best_params_
rid=Ridge()

best_clf = GridSearchCV(rid,scoring='neg_root_mean_squared_error',cv=10,n_jobs=1,

                        param_grid={'alpha': [0.01,0.1,1,10]})

best_clf.fit(df_train,train_y)
best_clf.best_score_
best_clf.best_params_
svr=SVR(kernel='poly',degree=2)

best_m=GridSearchCV(svr,param_grid={'C':[0.0001,0.001],'epsilon':[0.1,0.2,0.3],'gamma':[1,2]},

                   cv=10,n_jobs=3,scoring='neg_root_mean_squared_error')

best_m.fit(df_train,train_y)
best_m.best_params_
best_m.best_score_
reg=StackingRegressor(estimators=[('svm',SVR(kernel='poly',degree=2,C=0.0001,epsilon=0.1,gamma=1)),

                                 ('ridge',Ridge(alpha=10))],

                     final_estimator=PLSRegression(n_components=2))

cv=cross_validate(reg,X=df_train,y=train_y,cv=10,scoring='neg_root_mean_squared_error',n_jobs=1)
np.mean(cv['test_score'])
reg=StackingRegressor(estimators=[('svm',SVR(kernel='poly',degree=2,C=0.0001,epsilon=0.1,gamma=1)),

                                 ('ridge',Ridge(alpha=10))],

                     final_estimator=PLSRegression(n_components=2))

reg.fit(df_train,train_y)

y_pred=reg.predict(df_test)
y_pred=np.expm1(y_pred)

y_pred=pd.DataFrame(y_pred)

y_pred=y_pred.reset_index()

y_pred.columns=['Id','SalePrice']

y_pred['Id']=range(1461,2920)
submission=y_pred.copy()

q1 = submission['SalePrice'].quantile(0.005)

q2 = submission['SalePrice'].quantile(0.995)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)
submission.to_csv('./y_pred.csv',index=False)
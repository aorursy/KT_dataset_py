import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import r2_score,mean_squared_error,make_scorer
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
print(train.shape)
train.isnull().sum().sort_values(ascending=False)[0:20]
train.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'],axis=1,inplace=True)
train.isnull().sum().sort_values(ascending=False)[0:20]
plt.scatter(x=train['GrLivArea'],y=train['SalePrice'])
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.show()
train['GrLivArea'].sort_values(ascending=False)[:2]
train=train[(train['GrLivArea']<4500)]
plt.scatter(x=train['GrLivArea'],y=train['SalePrice'])
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.show()
sns.distplot(train['SalePrice'])
print('skewness:',train['SalePrice'].skew())
train_id=train['Id']
train.drop('Id',axis=1,inplace=True)
train.columns
train['MSSubClass'].value_counts()
train['MSZoning'].value_counts()
fig,ax=plt.subplots(figsize=(9,11))
sns.scatterplot(ax=ax,x=train['LotArea'],y=train['SalePrice'])
fig,ax=plt.subplots(figsize=(9,11))
sns.violinplot(ax=ax,x=train['OverallQual'],y=train['SalePrice'])
fig,ax=plt.subplots(figsize=(9,11))
sns.violinplot(ax=ax,x=train['OverallCond'],y=train['SalePrice'])
plt.scatter(data=train,x='YearBuilt',y='SalePrice')
cols=train[['TotalBsmtSF','GrLivArea','TotRmsAbvGrd','YrSold','OverallQual','GarageArea','SalePrice']]
sns.pairplot(cols)
sns.distplot(train['SalePrice'])
print('skewness before transforming:',train['SalePrice'].skew())
train['SalePrice']=np.log1p(train['SalePrice'])
sns.distplot(train['SalePrice'])
print('skewness after transform:',train['SalePrice'].skew())
train.isnull().sum().sort_values(ascending=False)[:30]
plt.scatter(data=train,x='SalePrice',y='LotFrontage')
plt.xlabel('sale price')
plt.ylabel('lotfrontage')
def impute_lot(cols):
    l=cols[0]
    s=cols[1]
    if pd.isnull(l):
        if s<=11.0:
            l=train[train['SalePrice']<=11.0]['LotFrontage'].mean()
            return l
        elif 11.0<s<=12.5:
            l=train[(train['SalePrice']>11.0)&(train['SalePrice']<=12.5)]['LotFrontage'].mean()
            return l
        else:
            l=train[(train['SalePrice']>12.5)]['LotFrontage'].mean()
            return l
    else:
        return l
train['LotFrontage']=train[['LotFrontage','SalePrice']].apply(impute_lot,axis=1)
train.isnull().sum().sort_values(ascending=False)[:15]
fig,ax=plt.subplots(figsize=(8,10))
sns.scatterplot(data=train,x='GarageYrBlt',y='SalePrice',hue='GarageType')
print(train['GarageType'].value_counts())
print(train[train['SalePrice']>11.8]['GarageType'].value_counts())
print(train[train['SalePrice']<=11.8]['GarageType'].value_counts())
def impute_gtype(cols):
    g=cols[0]
    s=cols[1]
    if pd.isnull(g):
        if s<=11.8:
            g='Detchd'
            return g
        else:
            g='Attchd'
            return g
    else:
        return g
train['GarageType']=train[['GarageType','SalePrice']].apply(impute_gtype,axis=1)
train.isnull().sum().sort_values(ascending=False)[:15]
corr_data=train.corr()
corr_data['GarageYrBlt'].sort_values(ascending=False)[:10]
sns.scatterplot(data=train,x='GarageYrBlt',y='YearBuilt')
sns.scatterplot(data=train,x='GarageYrBlt',y='YearBuilt')
plt.xlim(1900,)
plt.ylim(1900,)
def impute_gyear(cols):
    g=cols[0]
    y=cols[1]
    if pd.isnull(g):
        g=y
        return g
    else:
        return g
train['GarageYrBlt']=train[['GarageYrBlt','YearBuilt']].apply(impute_gyear,axis=1)
train.isnull().sum().sort_values(ascending=False)[:15]
train['GarageFinish'].value_counts()
fig,ax=plt.subplots(figsize=(8,10))
sns.scatterplot(data=train,x='GarageYrBlt',y='SalePrice',hue='GarageFinish')
print(train[train['SalePrice']<12]['GarageFinish'].value_counts())
print(train[(train['SalePrice']>12)&(train['SalePrice']<=12.5)]['GarageFinish'].value_counts())
print(train[train['SalePrice']>12.5]['GarageFinish'].value_counts())
def impute_gfinish(cols):
    g=cols[0]
    s=cols[1]
    if pd.isnull(g):
        if s<=12.0:
            g='Unf'
            return g
        elif (s>12.0)&(s<=12.5):
            g='RFn'
            return g
        else:
            g='Fin'
            return g
    else:
        return g
train['GarageFinish']=train[['GarageFinish','SalePrice']].apply(impute_gfinish,axis=1)
train.isnull().sum().sort_values(ascending=False)[:15]
train['GarageQual'].value_counts()
train['GarageQual'].fillna('TA',inplace=True)
train.isnull().sum().sort_values(ascending=False)[:10]
train['GarageCond'].value_counts()
train['GarageCond'].fillna('TA',inplace=True)
train.isnull().sum().sort_values(ascending=False)[:10]
train['BsmtFinType2'].value_counts()
train['BsmtFinType2'].fillna('Unf',inplace=True)
train.isnull().sum().sort_values(ascending=False)[:10]
train['BsmtExposure'].value_counts()
train['BsmtExposure'].fillna('No',inplace=True)
train.isnull().sum().sort_values(ascending=False)[:10]
train['BsmtQual'].value_counts()
sns.violinplot(data=train,y='SalePrice',x='BsmtQual')
print(train[train['SalePrice']>12]['BsmtQual'].value_counts())
print(train[train['SalePrice']<=12]['BsmtQual'].value_counts())
def impute_bqual(cols):
    b=cols[0]
    s=cols[1]
    if pd.isnull(b):
        if b<=12.0:
            b='TA'
            return b
        else:
            b='Gd'
            return b
    else:
        return b
train['BsmtQual']=train[['BsmtQual','SalePrice']].apply(impute_bqual,axis=1)
train.isnull().sum().sort_values(ascending=False)[:10]
train['BsmtCond'].value_counts()
train['BsmtCond'].fillna('TA',inplace=True)
train.isnull().sum().sort_values(ascending=False)[:10]
train['BsmtFinType1'].value_counts()
plt.subplots(figsize=(8,9))
sns.violinplot(data=train,y='SalePrice',x='BsmtFinType1')
print(train[train['SalePrice']>12]['BsmtFinType1'].value_counts())
print(train[train['SalePrice']<=12]['BsmtFinType1'].value_counts())
def impute_bfin1(cols):
    b=cols[0]
    s=cols[1]
    if pd.isnull(b):
        if b<=12.0:
            b='Unf'
            return b
        else:
            b='GLQ'
            return b
    else:
        return b
train['BsmtFinType1']=train[['BsmtFinType1','SalePrice']].apply(impute_bfin1,axis=1)
train.isnull().sum().sort_values(ascending=False)[:10]
train['MasVnrType'].fillna('None',inplace=True)
train['MasVnrArea'].fillna('None',inplace=True)
train['Electrical'].fillna('SBrkr',inplace=True)
train.isnull().sum().sort_values(ascending=False)[:10]
fig=plt.subplots(figsize=(9,8))
corr_data=train.corr()
sns.heatmap(corr_data)
train=pd.get_dummies(train)
train.shape
scorer=make_scorer(mean_squared_error,greater_is_better=False)
y=train['SalePrice']
train.drop('SalePrice',axis=1,inplace=True)
x_train,x_test,y_train,true_p=train_test_split(train,y,test_size=0.2,random_state=120)
print(x_train.shape,y_train.shape,x_test.shape,true_p.shape)
def rmse_cv_train(model):
    rmse=np.sqrt(-cross_val_score(model,x_train,y_train,scoring=scorer,cv=10))
    return rmse
def rmse_cv_test(model):
    rmse=np.sqrt(-cross_val_score(model,x_test,true_p,scoring=scorer,cv=10))
    return rmse
from sklearn.linear_model import LinearRegression,RidgeCV,LassoCV
lreg=LinearRegression()
lreg.fit(x_train,y_train)
print('rmse value of train data:',rmse_cv_train(lreg).mean())
print('rmse value of test data:',rmse_cv_test(lreg).mean())
train_pred=lreg.predict(x_train)
test_pred=lreg.predict(x_test)
plt.scatter(x=train_pred,y=train_pred-y_train,c='blue',marker='s',label='train data')
plt.scatter(x=test_pred,y=test_pred-true_p,c='green',marker='s',label='test data')
plt.xlabel('predicted values')
plt.ylabel('residuals')
plt.title('linear regression')
plt.plot([10.0,13.5],[0.0,0.0],c='red')
plt.show()


plt.scatter(x=train_pred,y=y_train,c='blue',marker='s',label='train data')
plt.scatter(x=test_pred,y=true_p,c='green',marker='s',label='test data')
plt.xlabel('predicted values')
plt.ylabel('real values')
plt.title('linear regression')
plt.plot([11,13.25],[11,13.25],c='red')
plt.show()

print('accuracy:',r2_score(test_pred,true_p))
alphas=[0.01,0.03,0.07,0.1,0.3,0.6,1,3,5,7,10,30,60]
ridge=RidgeCV(alphas)
ridge.fit(x_train,y_train)
alpha=ridge.alpha_
print('best alpha:',alpha)
alphas=[alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4]
ridge=RidgeCV(alphas,cv=10)
ridge.fit(x_train,y_train)
alpha=ridge.alpha_
print('optimised alphas:',alpha)
print('rmse value of train data:',rmse_cv_train(ridge).mean())
print('rmse value of test data:',rmse_cv_test(ridge).mean())
train_pred=ridge.predict(x_train)
test_pred=ridge.predict(x_test)
plt.scatter(x=train_pred,y=train_pred-y_train,c='blue',marker='s',label='train data')
plt.scatter(x=test_pred,y=test_pred-true_p,c='green',marker='s',label='test data')
plt.xlabel('predicted values')
plt.ylabel('residuals')
plt.title('ridge regression')
plt.plot([10.0,13.5],[0.0,0.0],c='red')
plt.show()


plt.scatter(x=train_pred,y=y_train,c='blue',marker='s',label='train data')
plt.scatter(x=test_pred,y=true_p,c='green',marker='s',label='test data')
plt.xlabel('predicted values')
plt.ylabel('real values')
plt.title('ridge regression')
plt.plot([11,13.25],[11,13.25],c='red')
plt.show()

print('accuracy:',r2_score(test_pred,true_p))
lasso = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 
                          0.3, 0.6, 1], 
                max_iter = 50000, cv = 10)
lasso.fit(x_train, y_train)
alpha = lasso.alpha_
print('best alpha:',alpha)
lasso = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, 
                          alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, 
                          alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35, 
                          alpha * 1.4], 
                max_iter = 50000, cv = 10)
lasso.fit(x_train, y_train)
alpha = lasso.alpha_
print('optimised alpha:',alpha)
print('rmse value of train data:',rmse_cv_train(lasso).mean())
print('rmse value of test data:',rmse_cv_test(lasso).mean())
train_pred=lasso.predict(x_train)
test_pred=lasso.predict(x_test)
plt.scatter(x=train_pred,y=train_pred-y_train,c='blue',marker='s',label='train data')
plt.scatter(x=test_pred,y=test_pred-true_p,c='green',marker='s',label='test data')
plt.xlabel('predicted values')
plt.ylabel('residuals')
plt.title('lasso regression')
plt.plot([10.0,13.5],[0.0,0.0],c='red')
plt.show()


plt.scatter(x=train_pred,y=y_train,c='blue',marker='s',label='train data')
plt.scatter(x=test_pred,y=true_p,c='green',marker='s',label='test data')
plt.xlabel('predicted values')
plt.ylabel('real values')
plt.title('lasso regression')
plt.plot([11,13.25],[11,13.25],c='red')
plt.show()

print('accuracy:',r2_score(test_pred,true_p))
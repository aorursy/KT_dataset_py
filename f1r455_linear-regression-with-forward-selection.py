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
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

n_train=train.shape[0]
y=train['SalePrice']
train=train.drop(['SalePrice'],axis=1)

data = pd.concat([train,test],ignore_index=True,sort=False)
train.head()
train.info()
nans=pd.isnull(data).sum()/data.shape[0]
print(nans[nans>0].sort_values(ascending=False)*100)
data = data.drop(['Id','PoolQC','MiscFeature','Alley','Fence','FireplaceQu'],axis=1)
data['LotFrontage'] = data[['LotFrontage','Neighborhood']].groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median()))
data['MSZoning'] = data[['MSZoning','MSSubClass']].groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()))
data.loc[:,['GarageQual','GarageCond','GarageFinish','GarageType']] = data.loc[:,['GarageQual','GarageCond','GarageFinish','GarageType']].fillna('None')
data['GarageYrBlt'] = data['GarageYrBlt'].fillna(0)
data = data.fillna(data.median())
print("Skewness of SalePrice: %f" %y.skew())
sns.distplot(y)
c_salePrice = pd.concat([data,y],axis=1).corr()['SalePrice']
c_salePrice = abs(c_salePrice).sort_values(ascending = False)

fig,ax=plt.subplots(figsize=(25,25))
sns.heatmap(pd.DataFrame(c_salePrice),annot=True,square=True,ax=ax,cmap='Blues')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
sns.regplot(train['OverallQual'],y)
sns.regplot(train['GrLivArea'],y)
print(data['Utilities'].value_counts())
print(data['Street'].value_counts())
print(data['Condition1'].value_counts())
print(data['Condition2'].value_counts())
print(data['BldgType'].value_counts())
data = data.drop(['Utilities','Street','Condition1','Condition2','BldgType'],axis=1)
def create_dummy(data,feature):
    dummy = pd.get_dummies(data[feature])
    dummy.columns=[feature+s for s in dummy.columns]
    data = pd.concat([data,dummy],axis=1)
    data=data.drop(feature,axis=1)
    return data
categorical=[f for f in data.columns if data.dtypes[f] =='object']
numeric=[f for f in data.columns if data.dtypes[f]!='object']

data[numeric]=data[numeric].astype('float64')
for f in categorical:
    data=create_dummy(data,f)
y=np.log1p(y)

data[numeric]=np.log1p(data[numeric])
scaler=RobustScaler()
data[numeric]=scaler.fit_transform(data[numeric])
train = data.iloc[:n_train,:]
test = data.iloc[n_train:,:]
test = test.reset_index()

def forward_selection(X,y):
    
    final_features=[]
    
    not_reduced_features=list(X.columns)
    MSE_hist=100000000000
    
    
    while (len(not_reduced_features)>0):
        MSE={}
        pvalue={}
        print('---------%d features ----------' %len(final_features))
        for f in not_reduced_features:
            features=final_features.copy()
            features.append(f)
            
            intercept=pd.DataFrame(np.ones((train.shape[0],1),dtype='float64'))
            intercept.columns=['Intercept']
            
            X_intercept = pd.concat([X[features],intercept],axis=1)
            
            reg = sm.OLS(endog=y,exog=X_intercept).fit()
            
            y_predict=reg.predict(X_intercept)
            r2=mean_squared_error(y,y_predict)
            
            MSE[f]=r2
            pvalue[f]=reg.pvalues[f]
                        
        mini,mini_f=min(zip(MSE.values(),MSE.keys()))
        if (mini<MSE_hist and pvalue[mini_f]<0.05):
            print('{} ---> MSE = {} ---- pvalue = {}\n'.format(mini_f,mini,pvalue[mini_f]))
            MSE_hist=mini
            final_features.append(mini_f)
            not_reduced_features.remove(mini_f)
        else:
            return final_features
    
    return final_features
            
            
features_selected = forward_selection(train,y)
train = train[features_selected]
test = test[features_selected]
intercept=pd.DataFrame(np.ones((train.shape[0],1),dtype='float64'))
intercept.columns=['Intercept']

train_intercept = pd.concat([train,intercept],axis=1)

# Train our model
reg=sm.OLS(y,train_intercept).fit()

# Predict the results
intercept=pd.DataFrame(np.ones((test.shape[0],1),dtype='float64'))
intercept.columns=['Intercept']

test_intercept = pd.concat([test,intercept],axis=1)

res=reg.predict(test_intercept)

# we make exp(x)-1 transformation : the reverse of log(1+x)
res=np.exp(res)-1

print(res[:9])
print(reg.summary())
print(reg.pvalues.sort_values())
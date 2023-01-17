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
import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')
train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
train.columns
train['SalePrice'].describe()
sns.distplot(train['SalePrice'])
print(train['SalePrice'].skew())#歪度

print(train['SalePrice'].kurt())#尖度
var='GrLivArea'

data=pd.concat([train['SalePrice'],train[var]],axis=1)

data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000));

#GrLivAreaとSalePriceとの相関関係を見る
var='TotalBsmtSF'

data=pd.concat([train['SalePrice'],train[var]],axis=1)

data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000))
var='OverallQual'

data=pd.concat([train['SalePrice'],train[var]],axis=1)

f,ax=plt.subplots(figsize=(8,6))

fig=sns.boxplot(x=var,y='SalePrice',data=data)

fig.axis(ylim=0,ymax=800000);
var = 'YearBuilt'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90);
corrmat=train.corr()

f,ax=plt.subplots(figsize=(12,9))

sns.heatmap(corrmat,vmax=.8,square=True);
k=10

cols=corrmat.nlargest(k,'SalePrice')['SalePrice'].index

cm=np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm=sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':10},yticklabels=cols.values,xticklabels=cols.values)

plt.show()
sns.set()

cols=['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt']

sns.pairplot(train[cols],size=3)

plt.show();
total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
train=train.drop((missing_data[missing_data['Total']>1]).index,1)

train=train.drop(train.loc[train['Electrical'].isnull()].index)

train.isnull().sum().max()
saleprice_scale=StandardScaler().fit_transform(train['SalePrice'][:,np.newaxis]);

low_range=saleprice_scale[saleprice_scale[:,0].argsort()][:10]

high_range=saleprice_scale[saleprice_scale[:,0].argsort()][-10:]



print('outer range low of the distribution')

print(low_range)

print('\outer range high of th distribution')

print(high_range)
var='GrLivArea'

data=pd.concat([train['SalePrice'],train[var]],axis=1)

data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000))
train.sort_values(by='GrLivArea',ascending=False)[:2]

train=train.drop(train[train['Id']==1299].index)

train=train.drop(train[train['Id']==524].index)
var = 'TotalBsmtSF'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
sns.distplot(train['SalePrice'],fit=norm);

fig=plt.figure()

res=stats.probplot(train['SalePrice'],plot=plt)
train['SalePrice']=np.log(train['SalePrice'])



sns.distplot(train['SalePrice'],fit=norm);

fig=plt.figure()

res=stats.probplot(train['SalePrice'],plot=plt)
train['GrLivArea']=np.log(train['GrLivArea'])



sns.distplot(train['GrLivArea'],fit=norm)

fig=plt.figure()

res=stats.probplot(train['GrLivArea'],plot=plt)
sns.distplot(train['TotalBsmtSF'],fit=norm)

fig=plt.figure()

res=stats.probplot(train['TotalBsmtSF'],plot=plt)
train['HasBsmt']=pd.Series(len(train['TotalBsmtSF']),index=train.index)

train['HasBsmt']=0

train.loc[train['TotalBsmtSF']>0,'HasBsmt']=1
train.loc[train['HasBsmt']==1,'TotalBsmtSF']=np.log(train['TotalBsmtSF'])
sns.distplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'],fit=norm);

fig=plt.figure()

res=stats.probplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'],plot=plt)
plt.scatter(train['GrLivArea'],train['SalePrice'])
plt.scatter(train[train['TotalBsmtSF']>0]['TotalBsmtSF'],train[train['TotalBsmtSF']>0]['SalePrice'])
from sklearn.preprocessing import StandardScaler



standard=StandardScaler()

train_std=pd.DataFrame(standard.fit_transform(train.loc[:,['TotalBsmtSF','GrLivArea','HasBsmt']]),columns=['TotalBsmtSF','GrLivArea','HasBsmt'])



train['TotalBsmtSF']=train_std['TotalBsmtSF']

train['GrLivArea']=train_std['GrLivArea']

train['HasBsmt']=train_std['HasBsmt']

train.head()
train.loc[:,['TotalBsmtSF','GrLivArea','HasBsmt']].isnull().sum()
train=train.fillna(train.mean())
train.loc[:,['TotalBsmtSF','GrLivArea','HasBsmt']].isnull().sum()
from sklearn.linear_model import LinearRegression 

from sklearn import preprocessing



lab_enc=preprocessing.LabelEncoder()



x_train=train.loc[:,['TotalBsmtSF','GrLivArea']].values

y_train=train.loc[:,['SalePrice']].values



y_train=lab_enc.fit_transform(y_train)

clf=LinearRegression()

clf.fit(x_train,y_train)
df_test=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_Id=df_test['Id']

df_test_index=df_test.loc[:,['SalePrice']]



df_test=df_test.loc[:,['TotalBsmtSF','GrLivArea']]

'''

df_test_std=pd.DataFrame(standard.transform(df_test.loc[:,['TotalBsmtSF','GrLivArea']]),columns=['TotalBsmtSF','GrLivArea'])



df_test['TotalBsmtSF']=df_test_std['TotalBsmtSF']

df_test['GrLivArea']=df_test_std['GrLivArea']

'''

df_test=df_test.fillna({'TotalBsmSF':0,'GrLivArea':0})



df_test.head()
x_test=df_test.values

y_test=clf.predict(x_test)
test=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

x_test=test.loc[:,['TotalBsmtSF','GrLivArea']]

predict=clf.predict(x_test)
predict=pd.DataFrame(predict)

predict.head()
predict=predict.rename(columns={0:'SalePrice'})

predict.head()
test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
result=pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

result['SalePrice']=predict
result.to_csv('result.csv',index=False)
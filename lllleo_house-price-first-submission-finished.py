import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns



import numpy as np



from sklearn.preprocessing import StandardScaler



from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')
# input data files

import pandas as pd



train = pd.read_csv('../input/train.csv')

test_full = pd.read_csv('../input/test.csv')
train.head()
test_full.head()
test_full.columns
train.columns
train['SalePrice'].describe()
import matplotlib.pyplot as plt



plt.hist(train['SalePrice'])
import seaborn as sns



sns.distplot(train['SalePrice'])

print ('Skewness:%f'% train['SalePrice'].skew())

print ('Kurtosis:%f'%train['SalePrice'].kurt())
data = train[['SalePrice','GrLivArea']]

data.plot.scatter(x='GrLivArea',y='SalePrice',ylim=(0,800000))
plt.scatter(data['GrLivArea'],data['SalePrice'])
data2 = train[['SalePrice','TotalBsmtSF']]

data2.plot.scatter(x='TotalBsmtSF',y='SalePrice',ylim=(0,800000))
data3=train[['SalePrice','OverallQual']]

f,ax=plt.subplots(figsize=(8,6))

fig=sns.boxplot(x='OverallQual',y='SalePrice',data=data3)

fig.axis(ymin=0,ymax=800000)
data4=train[['SalePrice','YearBuilt']]

f,ax=plt.subplots(figsize=(16,8))

fig=sns.boxplot(x='YearBuilt',y='SalePrice',data=data4)

fig.axis(ymin=0,ymax=800000)

plt.xticks(rotation=90)

corrmat = train.corr()

f,ax = plt.subplots(figsize=(12,9))

sns.heatmap(corrmat,vmax=8,square=True)
import numpy as np

k = 10

cols = corrmat.nlargest(k,'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm=sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':10})

plt.show()
cols
sns.set()

cols=['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt']

sns.pairplot(train[cols],size=2.5)

plt.show()
total = train.isnull().sum().sort_values(ascending=False)

percent=(train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data=pd.concat([total,percent],axis=1,keys=['Total','Percent'])

missing_data.head(20)
a = np.array(missing_data[missing_data['Total']>1].index)
li = []

for i in a:

    if i in train.columns:

        li.append(i)
li
train = train.drop(li,axis=1)

train = train.drop(train.loc[train['Electrical'].isnull()].index)

train.isnull().sum().max()
train.head()
from sklearn.preprocessing import StandardScaler

scaled = StandardScaler().fit_transform(train['SalePrice'][:,np.newaxis])

low_range=scaled[scaled[:,0].argsort()][:10]

high_range=scaled[scaled[:,0].argsort()][-10:]

print (low_range)

print (high_range)
d = train[['SalePrice','GrLivArea']]

d.plot.scatter(x='GrLivArea',y='SalePrice',ylim=(0,800000))
#deleting points

train.sort_values(by = 'GrLivArea', ascending = False)[:2]

train = train.drop(train[train['Id'] == 1299].index)

train = train.drop(train[train['Id'] == 524].index)
d2 = train[['SalePrice','TotalBsmtSF']]

d2.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0,800000));
train['SalePrice']=np.log(train['SalePrice'])

train = pd.get_dummies(train)
cols=['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt']

train = train[cols]
train.head()
from sklearn.linear_model import LinearRegression



train_target = train['SalePrice']

train_test = train[['OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt']]



regr=LinearRegression()

regr.fit(train_test,train_target)
test = test_full[['OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt']]

test.isnull().sum()

test['GarageCars'] = test['GarageCars'].fillna(value=0)

test['TotalBsmtSF']=test['TotalBsmtSF'].fillna(value=test['TotalBsmtSF'].mean())
y_pre = regr.predict(test[['OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt']])
result = pd.DataFrame({'Id':test_full['Id'],'SalePrice':y_pre})

result
# output result to csv file

result.to_csv('house_predict_price.csv',index=False)
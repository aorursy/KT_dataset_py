#import Library
%matplotlib inline
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
import copy
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# ignore Deprecation Warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

df_train = pd.read_csv('../input/train.csv')
df_test  = pd.read_csv('../input/test.csv')
df_train['SalePrice'].isnull().sum()
df_train.head()
#Explore missing value
missing = df_train.isnull().sum()
df_missing = pd.concat([missing,missing/len(df_train)*100], keys = ['count','percentage'],axis = 1).sort_values(by = 'percentage',ascending = False)
df_missing = df_missing[df_missing['count'] > 0]
df_missing
plt.figure(figsize = (10,7))
df_missing['percentage'].plot(kind = 'bar')
#Extract categorical and numerical variables
num = df_train.dtypes[df_train.dtypes != 'object'].index
cat = df_train.dtypes[df_train.dtypes == 'object'].index
cat
#Filling missing-value categorical
for i in cat:
    df_train[i].fillna('None',inplace = True)
    df_test[i].fillna('None',inplace = True)
df_train[cat].info()
#clear
#explore missing value on numerical feature
df_missing['percentage'][df_train[df_missing.index].dtypes != 'object'].sort_values(ascending = False).plot(kind ='bar')
plt.figure(figsize = (20,10))
sns.countplot(df_train['GarageYrBlt'])
select = df_train[df_train['GarageYrBlt'].isnull()]
select.head()
#fill GarageYrBlt with 0
df_train['GarageYrBlt'].fillna(0,inplace = True)
df_test['GarageYrBlt'].fillna(0,inplace = True)
df_train['MasVnrArea'].describe()
select = df_train[df_train['MasVnrArea'].isnull()]
select.head().T
#fill MasVnrArea with 0
df_train['MasVnrArea'].fillna(0,inplace = True)
df_test['MasVnrArea'].fillna(0,inplace = True)
missing_test = df_test.isnull().sum()
missing_test = missing_test[missing_test.values > 0]
missing_test
df_test[missing_test.index].dtypes
#Same as above
for i in missing_test.index:
    if (i !='LotFrontage'):
        df_test[i].fillna(0,inplace = True)
missing_test = df_test.isnull().sum()
missing_test = missing_test[missing_test.values > 0]
missing_test
length = len(df_train)
df_tr = df_train.drop('SalePrice',axis = 1)
df_reg = pd.concat([df_tr,df_test], axis = 0,ignore_index = True)
df_reg.tail()
#Forget to drop ID
df_reg.drop('Id',axis = 1,inplace = True)
df_train.drop('Id',axis = 1,inplace = True)
df_test.drop('Id',axis = 1,inplace = True)

#Do linear regression to predict LotFrontage (Ridge)
#Different with Lot-Frontage, it is really missing , so lets fix this with RLM
df_reg = pd.get_dummies(df_reg)
data_train = df_reg
data_train = data_train.dropna()
data_test = df_reg[df_reg.filter(like = 'LotFrontage').isnull().any(1)]
print(f'{len(data_train)} + {len(data_test)} = {len(df_reg)}')
X_train,y_train = data_train.drop('LotFrontage',axis = 1),data_train['LotFrontage']
y_test = data_test.drop('LotFrontage',axis=1)
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, Ridge, Lasso

for Model in [RidgeCV, LassoCV, ElasticNetCV]:
    model = Model()
    print('%s: %s' % (Model.__name__,
                      cross_val_score(model, X_train, y_train).mean()))
#Its likely RidgeCV do a better job
model = Ridge() #default-parameter
model.fit(X_train,y_train)
predict = model.predict(y_test)
predict
#return value to the original DataFrame
j = 0
#Data Train
row = df_train[df_train['LotFrontage'].isnull() == True].index
for i in row:
    df_train['LotFrontage'].iloc[i] = predict[j]
    j+=1
#Data test
row = df_test[df_test['LotFrontage'].isnull() == True].index
for i in row:
    df_test['LotFrontage'].iloc[i] = predict[j]
    j+=1

df_train.info()
df_test.info()
#Missing Value done, yea :"), now Focus on SalePrice !!!
#explore SalePrice
sns.distplot(df_train['SalePrice'])
#Quick- Overview heatmap
plt.figure(figsize = (15,10))
corr = df_train.corr()
sns.heatmap(corr)
#Top-10 correlation matrix
k = 10 #number of variables for heatmap
plt.figure(figsize = (10,10))
cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

#Quick-look at pair-plot
combine = ['SalePrice','OverallQual','GrLivArea','Neighborhood','ExterQual']
sns.pairplot(df_train[combine])
df_train.plot.scatter(x = 'GrLivArea', y = 'SalePrice')
#Erase the outlier
len(df_train)
df_train = df_train[df_train['GrLivArea'] < 4600]
df_train.plot.scatter(x = 'GrLivArea', y = 'SalePrice')
sns.boxplot(data = df_train, x = 'OverallQual', y = 'SalePrice')
plt.xticks(rotation = 90)
sns.boxplot(data = df_train, x = 'Neighborhood', y = 'SalePrice')
sns.boxplot(data = df_train, x = 'ExterQual', y = 'SalePrice')
#Back to Multicollinearity problem
sns.boxplot(data = df_train, x = 'GarageCars', y = 'GarageArea')
#Drop GarageArea
df_train.drop('GarageArea',axis = 1, inplace = True)
df_test.drop('GarageArea',axis = 1, inplace = True)
#Back to Multicollinearity problem
sns.boxplot(data = df_train, x = 'TotRmsAbvGrd', y = 'GrLivArea')
#Drop TotRmsAbvGrd
df_train.drop('TotRmsAbvGrd',axis = 1, inplace = True)
df_test.drop('TotRmsAbvGrd',axis = 1, inplace = True)
#Top-10 correlation matrix
corr = df_train.corr()
k = 10 #number of variables for heatmap
plt.figure(figsize = (10,10))
cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#Right-Skewed , not normal , lets see the others
from scipy.stats import skew
length = len(df_train)
df_all = pd.concat([df_train,df_test],axis = 0)
df_all['SalePrice'] = np.log1p(df_all['SalePrice'])
num = df_all.dtypes[df_all.dtypes != 'object'].index
for i in num:
    skw = skew(df_all[i])
    if (abs(skw) > 0.5):
        df_all[i] = np.log1p(df_all[i])
        print(f'{i} transformed!!')
#convert categorical to numerical
df_all = pd.get_dummies(df_all)
train = copy.copy(df_all[:length])
test  = copy.copy(df_all[length:])
X_train,y_train = train.drop('SalePrice',axis = 1),train['SalePrice']
y_test = test.drop('SalePrice',axis=1)
from sklearn.ensemble import RandomForestRegressor
for Model in [RidgeCV, LassoCV, ElasticNetCV, RandomForestRegressor]:
    model = Model()
    print('%s: %s' % (Model.__name__,
                      cross_val_score(model, X_train, y_train).mean()))
model = Ridge()
model.fit(X_train,y_train)
predict = model.predict(y_test)
predict
predict = np.expm1(predict)
predict
sub = pd.read_csv('../input/sample_submission.csv')
sub['SalePrice'] = predict
sub.to_csv('submission.csv',index=False)



























































































































z



































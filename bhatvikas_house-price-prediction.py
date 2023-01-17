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
import numpy as np
df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', keep_default_na=True,
                 na_values=['-1.#IND', '1.#QNAN',
                            '1.#IND', '-1.#QNAN','', '#N/A',       
                             'N/A',  '#NA', 'NULL', 'NaN', '-NaN', 'nan', '-nan'])
df.head(10)
df.drop(labels='Id',axis=1,inplace=True)
df['SalePrice'].describe()
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
print('skewness %f'%df['SalePrice'].skew())
print('Kurtosis %f'%df['SalePrice'].kurt())

plt.figure(figsize=(20,20))
test=df.corr()['SalePrice']
test.sort_values(ascending=False)
data = pd.concat([df['SalePrice'], df['OverallQual']], axis=1)
fig = sns.boxplot(x='OverallQual', y="SalePrice", data=data)
#plt.savefig('SalePricevsOverallQual')
data = pd.concat([df['SalePrice'], df['GarageCars']], axis=1)
fig = sns.boxplot(x='GarageCars', y="SalePrice", data=data)
plt.figure(figsize=(20,15))
sns.heatmap(df.corr(),annot =True)
df.isnull().sum()[df.isnull().sum()>0].sort_values(ascending=False)
plt.figure(figsize=(19,10))
sns.heatmap(data=df.isnull())
#plt.savefig('Missing_value')
# remove 'MiscFeature','PoolQC','Fence','Alley'... very large number of value are missing 
df=df.drop(labels=['MiscFeature','PoolQC','Fence','Alley','FireplaceQu','LotFrontage'],axis=1)
df.shape
plt.figure(figsize=(19,10))
sns.heatmap(data=df.isnull())
# few null are still remaning
df.isnull().sum()[df.isnull().sum()>0].sort_values(ascending=False)
#correlation with SalePrice
a=df.corr()['SalePrice']
a.sort_values(ascending=False)[:20]
df[a.sort_values(ascending=False)[:20].index].isnull().sum()
# fill most frequent year:
#Year garage was built
df['GarageYrBlt'].fillna(value=df['GarageYrBlt'].mode()[0],inplace=True)
df[a.sort_values(ascending=False)[:20].index].isnull().sum()
#Masonry veneer area in square feet
df['MasVnrArea'].fillna(value=df['MasVnrArea'].mode()[0],inplace=True)
df[a.sort_values(ascending=False)[:20].index].isnull().sum()
plt.figure(figsize=(19,10))
sns.heatmap(data=df.isnull())
#col which has null value
total_null=df.isnull().sum()
#All null value in decending order
null=df.isnull().sum()[total_null>0].sort_values(ascending=False)
# take index
index=null.index
# store in new file
new_null=df[index]
new_null.head()
sns.heatmap(new_null.isnull())
new_null.isnull().sum()
#Electrical only one null
sns.countplot(x='Electrical',data=df)
df['Electrical'].fillna(value='SBrkr',inplace=True)
df['MasVnrType'].fillna(value='None',inplace=True)
sns.countplot(x='BsmtQual',data=df)
col=['BsmtFinType1','BsmtCond','BsmtExposure','BsmtFinType2','BsmtQual'];
for i in col:
    df[i].fillna(value='NA',inplace=True)
sns.countplot(x='GarageCond',data=df)
col=['GarageCond','GarageQual','GarageFinish','GarageType'];
for i in col:
    df[i].fillna(value='NA',inplace=True)
null_value=df[df.isnull().sum()[df.isnull().sum()>0].sort_values(ascending=False).index]
null_value.head()
plt.figure(figsize=(19,10))
sns.heatmap(data=df.isnull())
plt.figure(figsize=(19,10))
sns.heatmap(data=df.isnull())
#Multicollinearity (remove)

df.drop(labels=["2ndFlrSF","1stFlrSF","GarageArea","TotRmsAbvGrd","GarageYrBlt"],axis=1,inplace=True)
plt.figure(figsize=(30,12))
sns.heatmap(df.drop(labels='SalePrice',axis=1).corr(),annot=True)
test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
test_df.shape
test_df.drop(labels='Id',axis=1,inplace=True)
#no first element is null
test_df[df.columns[:68]].head()
test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
test_df.shape
test_df.drop(labels='Id',axis=1,inplace=True)
df_u_col=df[df.columns[:68]].append(test_df[df.columns[:68]])
p=list(df_u_col['MSZoning'].unique())

for_nan =p[5]
for_nan
p.remove(for_nan)
p
#add test and train to find unique string in col
test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
test_df.shape
test_df.drop(labels='Id',axis=1,inplace=True)
df_u_col=df[df.columns[:68]].append(test_df[df.columns[:68]])


default_data={}
#find unique string
list_col=df.columns
for col in list_col:
    # only str colums

    if(isinstance(df[col].iloc[1], str)):
          print(col)
    else:
        continue
    #find unique string
    p=list(df_u_col[col].unique())
    #check if null value 
    if (df_u_col[col].isnull().any()):
        p.remove(for_nan)
    print(p)
    
    l=len(p)
    j=0
    for j in range(l):
        default_data.update({p.pop():j})
       
    #print(default_data)
    
    df[col]=df[col].map(default_data)
    default_data={}    



plt.figure(figsize=(19,10))
sns.heatmap(data=df.isnull())
df.isnull().sum()
df=df[df.corr().columns]
plt.figure(figsize=(19,10))
sns.heatmap(data=df.isnull())
print(df['SalePrice'].skew())
print(df['SalePrice'].kurt())
#df['SalePrice']=np.log(np.log(df['SalePrice']))

sns.distplot(np.log((df['SalePrice'])))
df['SalePrice']=np.log(df['SalePrice'])
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, confusion_matrix,classification_report
from sklearn.linear_model import  Ridge, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict

X=df.drop(labels='SalePrice',axis=1)
y=df['SalePrice']
#X=X[X.columns[1:20]]


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.20)


def model_eval(model):
    print(model)
    X=x_train
    y=y_train
    print('train--data')
    model_fit = model.fit(X, y)
    R2 = cross_val_score(model_fit, X, y, cv=10 , scoring='r2').mean()
    MSE = -cross_val_score(model_fit, X, y, cv=10 , scoring='neg_mean_squared_error').mean()
    print('R2 Score:', R2, '|', 'MSE:', MSE)
    X=x_test
    y=y_test
    print('test--data')
    y_pred=model.predict(X)
    R2 = cross_val_score(model_fit, X, y_test, cv=10 , scoring='r2').mean()
    MSE = -cross_val_score(model_fit, X, y, cv=10 , scoring='neg_mean_squared_error').mean()
    print('R2 Score:', R2, '|', 'MSE:', MSE)
    
    lr=LinearRegression()
ri = Ridge(alpha=0.1, normalize=False)
ricv = RidgeCV(cv=5)
gdb = GradientBoostingRegressor(n_estimators=300,learning_rate=0.1)

for model in [lr,ri,ricv,gdb]:
    print('model =={}'.format(model))
    model_eval(model)
    


print('on traning data')
#lr.fit(x_train,y_train)
y_pred=ri.predict(X)
score = cross_val_score(ri.fit(x_train,y_train), x_train, y_train, cv=10 , scoring='r2').mean()
print(score)
X=df.drop(labels='SalePrice',axis=1)
y=df['SalePrice']

gdb.fit(X,y)



gdb.predict(X)
gdb.feature_importances_
plt.figure(figsize=(30,12))
plt.bar(range(len(gdb.feature_importances_)), gdb.feature_importances_)
import xgboost
from xgboost import XGBRegressor
X=df.drop(labels='SalePrice',axis=1)
y=df['SalePrice']
#X=X[X.columns[1:20]]


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.20)
xgb = XGBRegressor(n_estimators=500,learning_rate=0.1)

model_eval(xgb)
from xgboost import plot_importance
f ,ax =plt.subplots(figsize=(30,30))

plot_importance(xgb,ax=ax,importance_type='weight')

x_test.columns
xgb.feature_importances_
importance =list(zip(x_test.columns,xgb.feature_importances_))
importance
importance.sort(key= lambda t : t[1])
importance
name , name2 =map(list,zip(*importance))
name
x_train=x_train[name]
x_test =x_test[name]
import xgboost
from xgboost import XGBRegressor
xgb = XGBRegressor(n_estimators=500,learning_rate=0.1)
X=x_train
y=y_train
print('train--data')
xgb = xgb.fit(X, y)
R2 = cross_val_score(xgb, X, y, cv=10 , scoring='r2').mean()
MSE = -cross_val_score(lr, X, y, cv=10 , scoring='neg_mean_squared_error').mean()
print('R2 Score:', R2, '|', 'MSE:', MSE)
X=x_test
y=y_test
print('test--data')
y_pred=xgb.predict(X)
R2 = cross_val_score(xgb, X, y_test, cv=10 , scoring='r2').mean()
MSE = -cross_val_score(xgb, X, y, cv=10 , scoring='neg_mean_squared_error').mean()
print('R2 Score:', R2, '|', 'MSE:', MSE)
print('full--data')
X=df.drop(labels='SalePrice',axis=1)[name]
y=df['SalePrice']
y_pred=xgb.predict(X)
R2 = cross_val_score(xgb, X, y, cv=10 , scoring='r2').mean()
MSE = -cross_val_score(xgb, X, y, cv=10 , scoring='neg_mean_squared_error').mean()
print('R2 Score:', R2, '|', 'MSE:', MSE)
from xgboost import plot_importance
f ,ax =plt.subplots(figsize=(30,30))
plot_importance(xgb,ax=ax)
xgb.feature_importances_
xgb.predict(X)
df['SalePrice'].isnull().sum()
test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
test_df.shape
test_df.drop(labels='Id',axis=1,inplace=True)
test_df=test_df.drop(labels=['MiscFeature','PoolQC','Fence','Alley','FireplaceQu','LotFrontage'],axis=1)
test_df.shape
# fill most frequent year:
#Year garage was built
test_df['GarageYrBlt'].fillna(value=test_df['GarageYrBlt'].mode()[0],inplace=True)
#tMasonry veneer area in square feet
test_df['MasVnrArea'].fillna(value=test_df['MasVnrArea'].mode()[0],inplace=True)
sns.heatmap(test_df.isnull())
#Multicollinearity (remove)
test_df.drop(labels=["GarageArea","TotRmsAbvGrd","2ndFlrSF","1stFlrSF","GarageYrBlt"],axis=1,inplace=True)
#test_df=test_df[test_df.corr().columns]

#test_df = pd.get_dummies(test_df)
sns.heatmap(test_df.isnull())
test_df.shape
X.shape
test_df['BsmtHalfBath'].fillna(value=test_df['BsmtHalfBath'].mode()[0],inplace=True)
test_df['GarageCars'].fillna(value=test_df['GarageCars'].mode()[0],inplace=True)
test_df['BsmtFullBath'].fillna(value=test_df['BsmtFullBath'].mode()[0],inplace=True)
test_df['BsmtFinSF1'].fillna(value=test_df['BsmtFinSF1'].mode()[0],inplace=True)
test_df['BsmtFinSF2'].fillna(value=test_df['BsmtFinSF2'].mode()[0],inplace=True)
test_df['BsmtUnfSF'].fillna(value=test_df['BsmtUnfSF'].mode()[0],inplace=True)
test_df['TotalBsmtSF'].fillna(value=test_df['TotalBsmtSF'].mode()[0],inplace=True)
a=test_df.isnull().sum()
a.sort_values(ascending=False)[:20]
col =['MasVnrType','MSZoning','Functional','Utilities','SaleType','Exterior1st','KitchenQual','Exterior2nd']
for col_i in col:
    test_df[col_i].fillna(value=test_df[col_i].mode()[0],inplace=True)
a=test_df.isnull().sum()
a.sort_values(ascending=False)[:20]
plt.figure(figsize=(22,8))

col=['GarageCond','GarageQual','GarageFinish','GarageType'];
for i in col:
    test_df[i].fillna(value='NA',inplace=True)
    col=['BsmtFinType1','BsmtCond','BsmtExposure','BsmtFinType2','BsmtQual'];
for i in col:
    test_df[i].fillna(value='NA',inplace=True)
sns.heatmap(test_df.isnull())
len(test_df.columns)
df =test_df
test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df_u_col=df[df.columns[:68]].append(test_df[df.columns[:68]])
default_data={}
#find unique string
list_col=df.columns


for col in list_col:
    # only str colums

    if(isinstance(df[col].iloc[1], str)):
          print(col)
    else:
        continue
    #find unique string
    p=list(df_u_col[col].unique())
    #check if null value 
    if (df_u_col[col].isnull().any()):
        p.remove(for_nan)
    print(p)
    
    l=len(p)
    j=0
    for j in range(l):
        default_data.update({p.pop():j})
       
    #print(default_data)
    
    df[col]=df[col].map(default_data)
    default_data={}    



plt.figure(figsize=(19,10))
sns.heatmap(data=df.isnull())
test_df=df
a=test_df.isnull().sum()
a.sort_values(ascending=False)[:20]
test_pred = model.predict(X)
test_df.shape
predicted_prices=np.exp(test_pred)
predicted_prices.reshape(1460)
for_id = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

submission = pd.DataFrame({'Id':for_id.Id, 'SalePrice': predicted_prices.reshape(1460)})
# you could use any filename. We choose submission here
submission.to_csv('submission1.csv', index=False)
test_pred =xgb.predict(test_df[name])
predicted_prices=np.exp(test_pred)
predicted_prices.reshape(1459)
for_id = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

my_submission = pd.DataFrame({'Id':for_id.Id, 'SalePrice': predicted_prices.reshape(1459)})
# you could use any filename. We choose submission here
my_submission.to_csv('submission_xgb.csv', index=False)
test_pred =gdb.predict(test_df)
predicted_prices=np.exp(test_pred)
predicted_prices.reshape(1459)
for_id = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

my_submission = pd.DataFrame({'Id':for_id.Id, 'SalePrice': predicted_prices.reshape(1459)})
# you could use any filename. We choose submission here
my_submission.to_csv('submission_gdb.csv', index=False)
test_pred =lr.predict(test_df)
predicted_prices=np.exp(test_pred)
predicted_prices.reshape(1459)
for_id = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

my_submission = pd.DataFrame({'Id':for_id.Id, 'SalePrice': predicted_prices.reshape(1459)})
# you could use any filename. We choose submission here
my_submission.to_csv('submission_lr.csv', index=False)
test_pred =ri.predict(test_df)
predicted_prices=np.exp(test_pred)
predicted_prices.reshape(1459)
for_id = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

my_submission = pd.DataFrame({'Id':for_id.Id, 'SalePrice': predicted_prices.reshape(1459)})
# you could use any filename. We choose submission here
my_submission.to_csv('submission_ri.csv', index=False)
test_pred =ricv.predict(test_df)
predicted_prices=np.exp(test_pred)
predicted_prices.reshape(1459)
for_id = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

my_submission = pd.DataFrame({'Id':for_id.Id, 'SalePrice': predicted_prices.reshape(1459)})
# you could use any filename. We choose submission here
my_submission.to_csv('submission_ricv.csv', index=False)
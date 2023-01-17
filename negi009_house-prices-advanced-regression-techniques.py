# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib  inline



import numpy as np

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
'''

import os 

import zipfile

# file name 

local_zip ='house-prices-advanced-regression-techniques.zip'

#Class with methods to open, read, write, close, list zip files.

zip_ref = zipfile.ZipFile(file=local_zip,mode='r')



zip_ref.extractall('house-price')

'''
#keep_default_na=False to prevent panda to interpreting na NaN , Na IS No Basement 

df = pd.read_csv('../input/train.csv', keep_default_na=True,

                 na_values=['-1.#IND', '1.#QNAN',

                            '1.#IND', '-1.#QNAN','', '#N/A',       

                             'N/A',  '#NA', 'NULL', 'NaN', '-NaN', 'nan', '-nan'])

df.head()
df.drop(labels='Id',axis=1,inplace=True)
df['SalePrice'].describe()

# check min >0
sns.distplot(df['SalePrice'])

# positive skewness
print('skewness %f'%df['SalePrice'].skew())

print('Kurtosis %f'%df['SalePrice'].kurt())
plt.figure(figsize=(20,20))

test=df.corr()['SalePrice']

test.sort_values(ascending=False)
#SalePriceis perpostional to OverallQual

data = pd.concat([df['SalePrice'], df['OverallQual']], axis=1)

fig = sns.boxplot(x='OverallQual', y="SalePrice", data=data)

#plt.savefig('SalePricevsOverallQual')
data = pd.concat([df['SalePrice'], df['GarageCars']], axis=1)

fig = sns.boxplot(x='GarageCars', y="SalePrice", data=data)
plt.figure(figsize=(12,8))

sns.heatmap(df.corr())
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
#y=df.isnull().sum()

#y2 = pd.DataFrame(data=y)

#y2[y2[0]>0]

#All null value in decending order

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
# only string type value has null

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
# 8 missing value  , ,Masonry veneer type

sns.countplot('MasVnrType',data=df)
sns.heatmap(new_null.isnull())
df['MasVnrType'].fillna(value='None',inplace=True)
#BsmtQual: Evaluates the height of the basement

#37 missing value

#Ex	Excellent (100+ inches)

# Gd	Good (90-99 inches)

# TA	Typical (80-89 inches)

# Fa	Fair (70-79 inches)

# Po	Poor (<70 inches

# NA	No Basement

sns.countplot(x='BsmtQual',data=df)


# no Basement same for BsmtFinType2    38

#BsmtExposure    38

#BsmtFinType1    37

#BsmtCond

col=['BsmtFinType1','BsmtCond','BsmtExposure','BsmtFinType2','BsmtQual'];

for i in col:

    df[i].fillna(value='NA',inplace=True)
sns.countplot(x='GarageCond',data=df)
sns.countplot(x='GarageQual',data=df)
col=['GarageCond','GarageQual','GarageFinish','GarageType'];

for i in col:

    df[i].fillna(value='NA',inplace=True)

    
#col which has null value()

null_value=df[df.isnull().sum()[df.isnull().sum()>0].sort_values(ascending=False).index]

null_value.head()

# no null value 

plt.figure(figsize=(19,10))

sns.heatmap(data=df.isnull())
plt.figure(figsize=(19,10))

sns.heatmap(data=df.isnull())
plt.figure(figsize=(30,12))

sns.heatmap(df.corr(),annot=True)
#Multicollinearity (remove)



df.drop(labels=["2ndFlrSF","1stFlrSF","GarageArea","TotRmsAbvGrd","GarageYrBlt"],axis=1,inplace=True)
plt.figure(figsize=(30,12))

sns.heatmap(df.drop(labels='SalePrice',axis=1).corr(),annot=True)
'''

df['MSZoning']=df['MSZoning'].map( {'A':0,'C':1,'FV':2,'I':3,'RH':4,'RL':5,'RP':6,'RM':7},na_action=True)



df['Street'] =df['Street'].map({'Grvl':0,'Pave':1})

df['LotShape']=df['LotShape'].map({'Reg':0,'IR1':1,'IR2':2,'IR3':3})

df['LandContour']= df['LandContour'].map({'Lvl':0,'Bnk':1,'HLS':2,'Low':3})

df['Utilities']=df['Utilities'].map({'AllPub':0,'NoSewr':1,'NoSeWa':2,'ELO':3})

df['LotConfig']=df['LotConfig'].map({'Inside':0,'Corner':1,'CulDSac':2,'FR2':3,'FR3':4})

''' 
test_df = pd.read_csv('../input/test.csv')

test_df.shape

test_df.drop(labels='Id',axis=1,inplace=True)

#no first element is null

test_df[df.columns[:68]].head()

test_df = pd.read_csv('../input/test.csv')

test_df.shape

test_df.drop(labels='Id',axis=1,inplace=True)

df_u_col=df[df.columns[:68]].append(test_df[df.columns[:68]])

p=list(df_u_col['MSZoning'].unique())



for_nan =p[5]

for_nan
p.remove(for_nan)

p
# add test and train to find unique string in col

test_df = pd.read_csv('../input/test.csv')

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
#df=pd.get_dummies(df)
df=df[df.corr().columns]

plt.figure(figsize=(19,10))

sns.heatmap(data=df.isnull())

print(df['SalePrice'].skew())

print(df['SalePrice'].kurt())


#df['SalePrice']=np.log(np.log(df['SalePrice']))



sns.distplot(np.log((df['SalePrice'])))

df['SalePrice']=np.log(df['SalePrice'])
#find unique string
print(df['SalePrice'].skew())

print(df['SalePrice'].kurt())
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
# WORKING ON NEURAL NETWORK 
import tensorflow as tf

model= tf.keras.models.Sequential(

    layers=[

        #input layer

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(units=10,activation=tf.nn.relu),

        tf.keras.layers.Dense(units=10,activation=tf.nn.relu),

        tf.keras.layers.Dense(units=10,activation=tf.nn.relu),

        tf.keras.layers.Dense(units=10,activation=tf.nn.relu),

        tf.keras.layers.Dense(units=10,activation=tf.nn.relu),

       tf.keras.layers.Dense(units=10,activation=tf.nn.relu),

        tf.keras.layers.Dense(units=10,activation=tf.nn.relu),

        tf.keras.layers.Dense(units=10,activation=tf.nn.relu),

       tf.keras.layers.Dense(units=10,activation=tf.nn.relu),

        

        # price

        

        

        tf.keras.layers.Dense(units=1,activation=tf.nn.relu)

    ]

)



model.compile(loss='mse', optimizer='adam',

              metrics=['mse'])
X=df.drop(labels='SalePrice',axis=1)

y=df['SalePrice']

hist=model.fit(x=np.array(X),y=np.array(y),batch_size=32,epochs=1000,verbose=2,validation_split=0.2,shuffle=True)



hist
y_predion=model.predict(np.array(X))
score = r2_score(y_true=np.array(y),y_pred=y_predion)

score
a=list(hist.params.values())

size=np.array(a)

size=size[1]

size
y_predion
fig = plt.figure(figsize=(25,12))

loss=hist.history['loss']

val_loss=hist.history['val_loss']

y1=np.arange(1,size+1)

plt.plot(y1,loss,'b',label='train')

plt.plot(y1,val_loss,'r',label='val')

plt.xlabel('epoch')

plt.ylabel('loss')

#plt.xlim(0,3900)

plt.legend()



plt.legend()
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
importance =importance[10:]
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
print(df['SalePrice'].skew())

print(df['SalePrice'].kurt())
df['SalePrice'].isnull().sum()
df.head()
test_df = pd.read_csv('../input/test.csv')

test_df.shape

test_df.drop(labels='Id',axis=1,inplace=True)
test_df=test_df.drop(labels=['MiscFeature','PoolQC','Fence','Alley','FireplaceQu','LotFrontage'],axis=1)
test_df.corr().index
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

sns.countplot('Exterior2nd',data=test_df)
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


# add test and train to find unique string in col

df =test_df

test_df = pd.read_csv('../input/train.csv')

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
test_pred = model.predict(test_df)

test_df.shape
predicted_prices=np.exp(test_pred)

predicted_prices.reshape(1459)
for_id = pd.read_csv('../input/test.csv')



my_submission = pd.DataFrame({'Id':for_id.Id, 'SalePrice': predicted_prices.reshape(1459)})

# you could use any filename. We choose submission here

my_submission.to_csv('submission_nn.csv', index=False)
test_pred =gdb.predict(test_df)

predicted_prices=np.exp(test_pred)

predicted_prices.reshape(1459)
for_id = pd.read_csv('../input/test.csv')



my_submission = pd.DataFrame({'Id':for_id.Id, 'SalePrice': predicted_prices.reshape(1459)})

# you could use any filename. We choose submission here

my_submission.to_csv('submission_gdb.csv', index=False)
test_pred =xgb.predict(test_df[name])

predicted_prices=np.exp(test_pred)

predicted_prices.reshape(1459)
for_id = pd.read_csv('../input/test.csv')



my_submission = pd.DataFrame({'Id':for_id.Id, 'SalePrice': predicted_prices.reshape(1459)})

# you could use any filename. We choose submission here

my_submission.to_csv('submission_xgb.csv', index=False)
test_pred =lr.predict(test_df)

predicted_prices=np.exp(test_pred)

predicted_prices.reshape(1459)
for_id = pd.read_csv('../input/test.csv')



my_submission = pd.DataFrame({'Id':for_id.Id, 'SalePrice': predicted_prices.reshape(1459)})

# you could use any filename. We choose submission here

my_submission.to_csv('submission_lr.csv', index=False)
test_pred =ri.predict(test_df)

predicted_prices=np.exp(test_pred)

predicted_prices.reshape(1459)
for_id = pd.read_csv('../input/test.csv')



my_submission = pd.DataFrame({'Id':for_id.Id, 'SalePrice': predicted_prices.reshape(1459)})

# you could use any filename. We choose submission here

my_submission.to_csv('submission_ri.csv', index=False)
test_pred =ricv.predict(test_df)

predicted_prices=np.exp(test_pred)

predicted_prices.reshape(1459)
for_id = pd.read_csv('../input/test.csv')



my_submission = pd.DataFrame({'Id':for_id.Id, 'SalePrice': predicted_prices.reshape(1459)})

# you could use any filename. We choose submission here

my_submission.to_csv('submission_ricv.csv', index=False)
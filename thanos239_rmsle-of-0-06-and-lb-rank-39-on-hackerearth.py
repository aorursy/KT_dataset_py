import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
train = pd.read_csv('../input/slashing-prices/Train.csv')
test=pd.read_csv('../input/slashing-prices/Test.csv')
train.head()
test.head()
train.tail()
test.tail()
print(test.nunique())
print()
print(train.nunique())
categorical_feats= ['Item_Id','Date']
numerical_feats = []
for col in train.columns:
    if col not in categorical_feats:
        numerical_feats.append(col)
numerical_feats
train[numerical_feats].describe()
numerical_feats_test=numerical_feats[:5]
numerical_feats_test.append(numerical_feats[-1])
test[numerical_feats_test].describe()
#checking correlation

num=train.select_dtypes(exclude='object')
numcorr=num.corr()
f,ax=plt.subplots(figsize=(17,1))
sns.heatmap(numcorr.sort_values(by=['Low_Cap_Price'], ascending=False).head(1), cmap='Reds')
plt.title(" Numerical features correlation with the sale price", weight='bold', fontsize=18)
plt.xticks(weight='bold')
plt.yticks(weight='bold', color='dodgerblue', rotation=0)
numcorr['Low_Cap_Price'].sort_values(ascending=False)
train['Date'].value_counts(sort=False).plot.bar(figsize=(15,5))
test['Date'].value_counts(sort=False).plot.bar(figsize=(15,5))
res = set.intersection(*(set(df['Product_Category']) for df in [train,test]))
res
train['Product_Category'].value_counts().plot.bar()
print('train')
test['Product_Category'].value_counts().plot.bar()
print('test')
train.groupby('Product_Category')['High_Cap_Price'].median().plot.bar()
train.groupby('Product_Category')['Low_Cap_Price'].median().plot.bar()
test.groupby('Product_Category')['High_Cap_Price'].median().plot.bar()
unique_train=train[numerical_feats[0]].unique()
unique_train.sort()
print(unique_train)
unique_test=test[numerical_feats[0]].unique()
unique_test.sort()
print(unique_test)
res = set.intersection(*(set(df[numerical_feats[0]]) for df in [train,test]))
res
train[numerical_feats[0]].value_counts().plot.bar()
print('Train')

test[numerical_feats[0]].value_counts().plot.bar()
print('Test')
train.groupby(numerical_feats[0])['High_Cap_Price'].median().plot.bar()
train.groupby(numerical_feats[0])['Low_Cap_Price'].median().plot.bar()
test.groupby(numerical_feats[0])['High_Cap_Price'].median().plot.bar()
train[numerical_feats[1]].value_counts(sort=False).plot.bar(figsize=(15,8))
print('train')
test[numerical_feats[1]].value_counts(sort=False).plot.bar(figsize=(15,8))
print('test')
train[numerical_feats[3]].value_counts(sort=False).plot.bar()
print('train')
test[numerical_feats[3]].value_counts(sort=False).plot.bar()
print('test')
train[numerical_feats[4]].value_counts(sort=False).plot.bar(figsize=(15,5))
print(numerical_feats[4])
train.groupby('Product_Category')['Demand'].mean().plot.bar()
train.groupby('State_of_Country')['Demand'].mean().plot.bar()
plt.figure(figsize=(15,6))
plt.scatter(x=train['High_Cap_Price'], y=train['Low_Cap_Price'], color='crimson', alpha=0.5)
plt.title('High/low', weight='bold', fontsize=16)
plt.xlabel('High Price', weight='bold', fontsize=12)
plt.ylabel('Low Price', weight='bold', fontsize=12)
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.show()
plt.figure(figsize=(15,6))
plt.scatter(x=train['Demand'], y=train['Low_Cap_Price'], color='crimson', alpha=0.5)
plt.title('Demand/low', weight='bold', fontsize=16)
plt.xlabel('Demand', weight='bold', fontsize=12)
plt.ylabel('Low Price', weight='bold', fontsize=12)
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.axis([0, 50,0,20000])
plt.show()
#Let's set the threshold as 40(this is totally experimental), and categorize as low demand and high demand.
train['low_high']=train['Demand']>=40
test['low_high']=test['Demand']>=40
train['low_high'].value_counts().plot.bar()
test['low_high'].value_counts().plot.bar()
from datetime import datetime  
train['Date'] = pd.to_datetime(train.Date,format='%Y-%m-%d') 
test['Date'] = pd.to_datetime(test.Date,format='%Y-%m-%d')
for i in (train, test):
    i['year']=i.Date.dt.year 
    i['month']=i.Date.dt.month 
    i['day']=i.Date.dt.day
   
train['day of week']=train['Date'].dt.dayofweek 
temp = train['Date']
test['day of week']=test['Date'].dt.dayofweek 
def applyer(row):
    if row.dayofweek ==4 or row.dayofweek == 5 or row.dayofweek == 6:
        return 1
    else:
        return 0 
temp2 = train['Date'].apply(applyer) 
train['weekend']=temp2
temp3 = test['Date'].apply(applyer) 
test['weekend']=temp3
num=train.select_dtypes(exclude='object')
numcorr=num.corr()
f,ax=plt.subplots(figsize=(17,1))
sns.heatmap(numcorr.sort_values(by=['Low_Cap_Price'], ascending=False).head(1), cmap='Reds')
plt.title(" Numerical features correlation with the sale price", weight='bold', fontsize=18)
plt.xticks(weight='bold')
plt.yticks(weight='bold', color='dodgerblue', rotation=0)
numcorr['Low_Cap_Price'].sort_values(ascending=False)
train.head()
train.drop(['Date','Demand','year','Item_Id', 'day of week', 'day'],axis=1,inplace=True)
test.drop(['Date','Demand','year','Item_Id', 'day of week', 'day'],axis=1,inplace=True)
from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()
scaler.fit(train.drop(['Low_Cap_Price'],axis=1))
scaled_features_train= scaler.transform(train.drop('Low_Cap_Price',axis=1))
from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()
scaler.fit(test)
scaled_features_train= scaler.transform(test)
train.info()
from sklearn.model_selection import train_test_split, cross_val_score
X= train[['State_of_Country','Market_Category','Product_Category','Grade','High_Cap_Price','low_high','month','weekend']]
y= train['Low_Cap_Price'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=101)
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
models=[]
models.append(('DTC',DecisionTreeRegressor()))
models.append(('KNC',KNeighborsRegressor()))
#models.append(('LR',LinearRegression()))
models.append(('RFC',RandomForestRegressor()))
#models.append(("MLP",MLPRegressor()))
models.append(("GBC",GradientBoostingRegressor()))
names=[]
for name,algo in models:
    algo.fit(X_train,y_train)
    prediction= algo.predict(X_test)
    a= metrics.mean_squared_log_error(y_test,prediction) 
    print("%s: %f "%(name, a))
rm= RandomForestRegressor(random_state=22, n_estimators=400)
rm.fit(X_train,y_train)
X_= test[['State_of_Country','Market_Category','Product_Category','Grade','High_Cap_Price','low_high','month','weekend']]
prediction = rm.predict(X_)
rm2= RandomForestRegressor(random_state=101, n_estimators=400)
rm2.fit(X_train, y_train)
prediction2= rm2.predict(X_)
gb= GradientBoostingRegressor(random_state=101, n_estimators=400)
gb.fit(X_train,y_train)
pred=gb.predict(X_)
gb2= GradientBoostingRegressor(random_state=101, n_estimators=400)
gb2.fit(X_train,y_train)
pred2=gb2.predict(X_)
gb3= GradientBoostingRegressor(random_state=101, n_estimators=400)
gb3.fit(X_train,y_train)
pred3=gb3.predict(X_)
gb4= GradientBoostingRegressor()
gb4.fit(X_train,y_train)
pred4=gb4.predict(X_)
gb5= GradientBoostingRegressor()
gb5.fit(X_train,y_train)
pred5=gb5.predict(X_)
pre=(prediction+prediction2+pred+pred2+pred3+pred4+pred5)/7
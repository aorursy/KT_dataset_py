import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
data = pd.read_csv('../input/train_V2.csv')

datanew = data.copy()

data2 = pd.read_csv('../input/test_V2.csv')
data.head()
data = data.iloc[:50000,:]
data.isnull().sum()
data1 = data.copy()
data1.drop(['Id','groupId','matchId','matchType'],inplace=True,axis=1)

data2.drop(['groupId','matchId','matchType'],inplace=True,axis=1)
f,ax = plt.subplots(figsize=(15,15))

ax = sns.heatmap(data1.corr(),annot=True,fmt= '.1f',linewidths=.5)

plt.show()
m = data1.corr()['winPlacePerc']

m.sort_values(ascending=False).head(6)
f,ax = plt.subplots(figsize=(15,8))

sns.barplot(data.matchType.value_counts().values,data.matchType.value_counts().index)

plt.xlabel('Count')

plt.ylabel('matchType')

plt.show()
d = data1.assists.value_counts()

d = d[d.index>0]
f,ax = plt.subplots(figsize=(15,5))

ax = sns.barplot(d.index,d.values,palette='rainbow')

plt.xlabel('Assists')

plt.ylabel('Count')

plt.show()
f,ax = plt.subplots(figsize=(15,4))

ax = sns.distplot(data1.DBNOs,color='green')

plt.show()
f,ax = plt.subplots(figsize=(15,4))

ax = sns.distplot(data1.matchDuration,color='green')

plt.show()
d = data1.kills.value_counts()

d = d[d.index>0]
f,ax = plt.subplots(figsize=(15,4))

ax = sns.barplot(d.index,d.values,palette = 'coolwarm')

plt.xlabel('kills')

plt.ylabel('count')

plt.show()
sns.jointplot(x ='winPlacePerc',y='walkDistance',data = data1,height= 10,color='purple')

plt.show()
ax = sns.jointplot(x ='winPlacePerc',y='boosts',data = data1,height=10,color='green')

plt.show()
ax = sns.jointplot(data1.winPlacePerc,data1.weaponsAcquired,height=10)

plt.show()
ax = sns.jointplot(data1.winPlacePerc,data1.damageDealt,height = 10,color ='violet')

plt.show()
ax = sns.jointplot(data1.winPlacePerc,data1.heals,height= 10,color='violet')

plt.show()
X = data1.iloc[:,:24].values

Y = data1.iloc[:,-1].values

X_ = data2.iloc[:,1:].values
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.30)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()  

X_train = sc.fit_transform(X_train)  

X_test = sc.transform(X_test) 
from sklearn.linear_model import LinearRegression

lin = LinearRegression()

lin.fit(X_train,Y_train)

lin.score(X_test,Y_test)
from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor(max_depth = 10,criterion='mse',n_estimators=100)

reg.fit(X_train,Y_train)
Y_pred = reg.predict(X_test)
from sklearn.metrics import r2_score

sc = r2_score(Y_test,Y_pred)

sc
d3 = pd.DataFrame()

test_pred = reg.predict(X_)

d3['Id'] = data2['Id']

d3['winPlacePerc'] = test_pred

d3.to_csv('sample_submission_V2.csv',index=False)
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

import matplotlib  
import matplotlib.pyplot as plt 
train = pd.read_csv('/kaggle/input/sf-crime/train.csv.zip')
train.shape
train.head()
train.info()
train['Dates']=pd.to_datetime(train['Dates'])
train['Year'] = train['Dates'].dt.year
train['QTR'] = train['Dates'].dt.quarter
train['Month'] = train['Dates'].dt.month
train['Hour'] = train['Dates'].dt.hour
daynumber = {
    'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7
}
train['DayOfWeek'] = train['DayOfWeek'].map(daynumber)

plot_data = train.groupby(['Month','Hour']).agg({'Category':'count'}).\
pivot_table(index='Hour',columns='Month')['Category']
fig = plt.figure() 
ax = fig.add_subplot() 
x = plot_data.index 
for col in plot_data.columns:
    y = plot_data[col]
    ax.plot(x,y,'o-',label=col)
ax.legend(title='Month') 
ax.grid('both',alpha=0.3)
ax.set_xlabel('Hour')
plt.show()
plot_data = train.groupby(['Year','Month']).agg({'Category':'count'}).\
pivot_table(index='Month',columns='Year')['Category']
fig = plt.figure() 
ax = fig.add_subplot() 
x = plot_data.index 
for col in plot_data.columns:
    y = plot_data[col]
    ax.plot(x,y,'o-',label=col)
ax.legend(title='Year') 
ax.grid('both',alpha=0.3)
ax.set_xlabel('Month')
plt.show()
plot_data = train.groupby(['DayOfWeek']).agg({'Category':'count'})
fig = plt.figure() 
ax = fig.add_subplot() 
x = plot_data.index 
for col in plot_data.columns:
    y = plot_data[col]
    ax.plot(x,y,'o-',label=col)
# ax.legend(title='Year') 
ax.grid('both',alpha=0.3)
ax.set_xlabel('DayOfWeek')
plt.show()
fig = plt.figure(figsize=(20,20)) 
ax = fig.add_subplot() 
for dist in train['PdDistrict'].unique():
    idx = train['PdDistrict'] == dist
    df2 = train[idx].copy() 
    x = df2['X']
    y = df2['Y']
    ax.scatter(x,y,label=dist)
ax.legend()
# 이상치 해결
train2 = train[(train['X'] < -121)]
XYS = train2.groupby(['PdDistrict']).agg({'X':np.mean,'Y':np.mean})
for i in train.index:
    v = train.loc[i,'X']
    d = train.loc[i,'PdDistrict']
    if v > -121:
        train.loc[i,['X','Y']] = XYS.loc[d,['X','Y']]
fig = plt.figure(figsize=(20,20)) 
ax = fig.add_subplot() 
for dist in train['PdDistrict'].unique():
    idx = train['PdDistrict'] == dist
    df2 = train[idx].copy() 
    x = df2['X']
    y = df2['Y']
    ax.scatter(x,y,label=dist)
ax.legend()
dist = train['PdDistrict'].unique()[4]
idx = train['PdDistrict'] == dist 
df2 = train[idx].copy() 
plot_data = df2.groupby(['Year','Month']).agg({'Category':'count'}).\
pivot_table(index='Month',columns='Year')['Category']

fig = plt.figure() 
ax = fig.add_subplot() 
x = plot_data.index 
for col in plot_data.columns:
    y = plot_data[col]
    ax.plot(x,y,'o-',label=col)
# ax.legend(title='Year') 
ax.grid('both',alpha=0.3)
ax.set_xlabel('Month')
ax.set_title(dist)
plt.show()
train.groupby(['PdDistrict']).agg({'Category':'count'}).plot(kind='barh')
train.groupby(['PdDistrict','Category']).\
agg({'Resolution':'count'}).\
pivot_table(index='Category',columns='PdDistrict')['Resolution']
train.groupby(['PdDistrict','Category']).\
agg({'Resolution':'count'}).\
pivot_table(index='Category',columns='PdDistrict')['Resolution'].plot(kind='barh',stacked=True,figsize=(5,20))
ct2 = train['Category'].value_counts()[train['Category'].value_counts() > 10000].index
train['Category2']=train['Category'].apply(lambda x: 'Others' if x not in ct2 else x)
train['Category2'].unique()

train.groupby(['PdDistrict','Category2']).\
agg({'Resolution':'count'}).\
pivot_table(index='PdDistrict',columns='Category2')['Resolution'].plot(kind='bar',stacked=True,figsize=(10,8))
df3 = train.groupby(['PdDistrict','Category2']).\
agg({'Resolution':'count'}).\
pivot_table(index='PdDistrict',columns='Category2')['Resolution']
ttl = df3.sum(axis=1)
ratio = df3.copy() 
for col in df3.columns:
    ratio[col] = df3[col]/ttl*100
ratio.plot(kind='bar',stacked=True,figsize=(10,8))
import pandas as pd 
import numpy as np 
train = pd.read_csv('/kaggle/input/sf-crime/train.csv.zip')
train['Dates']=pd.to_datetime(train['Dates'])
train['Hour'] = train['Dates'].dt.hour
# 이상치 제거
train2 = train[(train['X'] < -121)]
XYS = train2.groupby(['PdDistrict']).agg({'X':np.mean,'Y':np.mean})
for i in train.index:
    v = train.loc[i,'X']
    d = train.loc[i,'PdDistrict']
    if v > -121:
        train.loc[i,['X','Y']] = XYS.loc[d,['X','Y']]
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train["Category"] = le.fit_transform(train["Category"])
le = LabelEncoder()
train["PdDistrict"] = le.fit_transform(train["PdDistrict"])
le = LabelEncoder()
train["DayOfWeek"] = le.fit_transform(train["DayOfWeek"])
train3 = train[:500000].copy() 
# 학습할 모델
train2 = train3[['Category','DayOfWeek','PdDistrict','X','Y','Hour']].copy() 
X = train2.drop("Category",axis=1).values
y = train2["Category"].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
from sklearn.neighbors  import KNeighborsClassifier 
for n in np.arange(5,101,10):
    model = KNeighborsClassifier(n_neighbors=n) 
    # X_train과 y_train으로 학습을 하고 
    model.fit(X_train, y_train)
    # X_vaild와 y_vaild로 예측을 연습 
    pred_train = model.predict(X_test)
    score = (pred_train == y_test).mean()
    print('n_neighbors = {} : '.format(n)+str(score))
# 범죄의 유형이 너무 많아서 예측하기 어려운 것일까 ? 
import pandas as pd 
import numpy as np 
train = pd.read_csv('/kaggle/input/sf-crime/train.csv.zip')
train['Dates']=pd.to_datetime(train['Dates'])
train['Hour'] = train['Dates'].dt.hour
# 이상치 제거
train2 = train[(train['X'] < -121)]
XYS = train2.groupby(['PdDistrict']).agg({'X':np.mean,'Y':np.mean})
for i in train.index:
    v = train.loc[i,'X']
    d = train.loc[i,'PdDistrict']
    if v > -121:
        train.loc[i,['X','Y']] = XYS.loc[d,['X','Y']]

ct2 = train['Category'].value_counts()[train['Category'].value_counts() > 10000].index
train['Category2']=train['Category'].apply(lambda x: 'Others' if x not in ct2 else x)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train["Category"] = le.fit_transform(train["Category"])
le = LabelEncoder()
train["PdDistrict"] = le.fit_transform(train["PdDistrict"])
le = LabelEncoder()
train["DayOfWeek"] = le.fit_transform(train["DayOfWeek"])

train3 = train[:500000].copy() 
# 학습할 모델
train2 = train3[['Category2','DayOfWeek','PdDistrict','X','Y','Hour']].copy() 
X = train2.drop("Category2",axis=1).values
y = train2["Category2"].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

from sklearn.neighbors  import KNeighborsClassifier 
for n in np.arange(10,101,10):
    model = KNeighborsClassifier(n_neighbors=n) 
    # X_train과 y_train으로 학습을 하고 
    model.fit(X_train, y_train)
    # X_vaild와 y_vaild로 예측을 연습 
    pred_train = model.predict(X_test)
    score = (pred_train == y_test).mean()
    print('n_neighbors = {} : '.format(n)+str(score))
    
test = pd.read_csv('./test.csv')
test['Dates']=pd.to_datetime(test['Dates'])
test['Hour'] = test['Dates'].dt.hour
# 이상치 제거
train2 = test[(test['X'] < -121)]
XYS = train2.groupby(['PdDistrict']).agg({'X':np.mean,'Y':np.mean})
for i in train.index:
    v = test.loc[i,'X']
    d = test.loc[i,'PdDistrict']
    if v > -121:
        test.loc[i,['X','Y']] = XYS.loc[d,['X','Y']]
from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# test["Category"] = le.fit_transform(test["Category"])
le = LabelEncoder()
test["PdDistrict"] = le.fit_transform(test["PdDistrict"])
le = LabelEncoder()
test["DayOfWeek"] = le.fit_transform(test["DayOfWeek"])
test2 = test.loc[:20000,['DayOfWeek','PdDistrict','X','Y','Hour']].copy() 
result = model.predict(test2)
result
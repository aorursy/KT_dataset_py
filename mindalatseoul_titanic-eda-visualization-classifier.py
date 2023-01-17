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
train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.columns
# 데이터 시각화 패키지
import matplotlib 
import matplotlib.pyplot as plt  
matplotlib.rc('font',family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False
# 가설1 : 티켓 등급별로 생존율 차이가 있다  
df = train.groupby(['Pclass']).agg({'Survived':[np.sum,'count']})['Survived']
df['ratio'] = round(df['sum']/df['count']*100,1)
plot_data = df.copy() 
fig = plt.figure(facecolor='white',figsize=(10,4)) 
ax = fig.add_subplot() 
x = plot_data.index 
y = plot_data['ratio']
ax.bar(x,y)
ax.grid(axis='y',alpha=0.3)
plt.show()
# 가설2 : 성별에 따라 생존율 차이가 있다 
df = train.groupby(['Sex']).agg({'Survived':[np.sum,'count']})['Survived']
df['ratio'] = round(df['sum']/df['count']*100,1)
plot_data = df.copy() 
fig = plt.figure(facecolor='white',figsize=(10,4)) 
ax = fig.add_subplot() 
x = plot_data.index 
y = plot_data['ratio']
ax.bar(x,y)
ax.grid(axis='y',alpha=0.3)
plt.show()
# 가설3 : 나이에 따라 생존율 차이가 있다 
# 먼저 나이를 하나의 범주로 그룹핑
for i in train.index:
    age = train.loc[i,'Age']
    if age < 20:
        train.loc[i,'AgeGroup'] = '0~19'
    elif (age >= 20) & (age < 40):
        train.loc[i,'AgeGroup'] = '20~39'
    elif (age >= 40) & (age < 80):
        train.loc[i,'AgeGroup'] = '40~79'
    elif age >= 80:
        train.loc[i,'AgeGroup'] = '80~'
    else:
        continue

df = train.groupby(['AgeGroup']).agg({'Survived':[np.sum,'count']})['Survived']
df['ratio'] = round(df['sum']/df['count']*100,1)
plot_data = df.copy() 
fig = plt.figure(facecolor='white',figsize=(10,4)) 
ax = fig.add_subplot() 
x = plot_data.index 
y = plot_data['ratio']
ax.bar(x,y)
ax.grid(axis='y',alpha=0.3)
plt.show()
# 가설4 : 티켓 가격에 따라 생존율 차이가 있다 
# 먼저 티켓 가격을 하나의 범주로 그룹핑
for i in train.index:
    fare = train.loc[i,'Fare']
    if fare < 100:
        train.loc[i,'FareGroup'] = '0~99'
    elif (fare >= 100) & (fare < 200):
        train.loc[i,'FareGroup'] = '100~199'
    elif (fare >= 200) & (fare < 300):
        train.loc[i,'FareGroup'] = '200~299'
    elif (fare >= 300) & (fare < 400):
        train.loc[i,'FareGroup'] = '300~399'
    elif fare >= 400:
        train.loc[i,'FareGroup'] = '400~'
    else:
        continue

df = train.groupby(['FareGroup']).agg({'Survived':[np.sum,'count']})['Survived']
df['ratio'] = round(df['sum']/df['count']*100,1)
plot_data = df.copy() 
fig = plt.figure(facecolor='white',figsize=(10,4)) 
ax = fig.add_subplot() 
x = plot_data.index 
y = plot_data['ratio']
ax.bar(x,y)
ax.grid(axis='y',alpha=0.3)
plt.show()
# 가설5 : 승선 항구에 따라 생존율 차이가 있다 
df = train.groupby(['Embarked']).agg({'Survived':[np.sum,'count']})['Survived']
df['ratio'] = round(df['sum']/df['count']*100,1)
plot_data = df.copy() 
fig = plt.figure(facecolor='white',figsize=(10,4)) 
ax = fig.add_subplot() 
x = plot_data.index 
y = plot_data['ratio']
ax.bar(x,y)
ax.grid(axis='y',alpha=0.3)
plt.show()
# 가설6 : 티켓 등급과 성별에 따라서 생존율 차이가 있다 
df = train.groupby(['Sex','Pclass']).agg({'Survived':[np.sum,'count']})['Survived']
df['ratio'] = round(df['sum']/df['count']*100,1)
plot_data = df.copy() 
plot_data
fig = plt.figure(facecolor='white',figsize=(10,4)) 
ax = fig.add_subplot() 
x_lbl = plot_data.index 
x = np.arange(len(x_lbl))
y = plot_data['ratio']
ax.bar(x,y)
ax.set_xticks(x)
ax.set_xticklabels(str(x1) for x1 in x_lbl)
ax.grid(axis='y',alpha=0.3)
plt.show()
# 가설7 : 성별과 나이에 따라서 생존율 차이가 있다 
df = train.groupby(['Sex','AgeGroup']).agg({'Survived':[np.sum,'count']})['Survived']
df['ratio'] = round(df['sum']/df['count']*100,1)
plot_data = df.copy() 
plot_data
fig = plt.figure(facecolor='white',figsize=(10,4)) 
ax = fig.add_subplot() 
x_lbl = plot_data.index 
x = np.arange(len(x_lbl))
y = plot_data['ratio']
ax.barh(x,y)
ax.set_yticks(x)
ax.set_yticklabels(str(x1) for x1 in x_lbl)
ax.grid(axis='y',alpha=0.3)
plt.show()
# 가설8 : 승선항구와 티켓등급에 따라서 생존율 차이가 있다 
df = train.groupby(['Embarked','Pclass']).agg({'Survived':[np.sum,'count']})['Survived']
df['ratio'] = round(df['sum']/df['count']*100,1)
plot_data = df.copy() 
plot_data
fig = plt.figure(facecolor='white',figsize=(10,4)) 
ax = fig.add_subplot() 
x_lbl = plot_data.index 
x = np.arange(len(x_lbl))
y = plot_data['ratio']
ax.barh(x,y)
ax.set_yticks(x)
ax.set_yticklabels(str(x1) for x1 in x_lbl)
ax.grid(axis='y',alpha=0.3)
plt.show()
# 가설9 : 동행인 수에 따라서 생존율 차이가 있다 
df = train.groupby(['SibSp','Parch']).agg({'Survived':[np.sum,'count']})['Survived']
df['ratio'] = round(df['sum']/df['count']*100,1)
plot_data = df.copy() 
plot_data
fig = plt.figure(facecolor='white',figsize=(10,8)) 
ax = fig.add_subplot() 
x_lbl = plot_data.index 
x = np.arange(len(x_lbl))
y = plot_data['ratio']
ax.barh(x,y)
ax.set_yticks(x)
ax.set_yticklabels(str(x1) for x1 in x_lbl)
ax.grid(axis='y',alpha=0.3)
plt.show()
print('정제 전 결측값')
print(train.isna().sum())
idx = ~train['Age'].isna()
df = train.loc[idx,:].copy() 
df2 = df.groupby(['Embarked','Pclass','Sex']).agg({'Age':np.mean})

idx = train['Age'].isna() 
for i in train[idx].index:
    e = train.loc[i,'Embarked']
    p = train.loc[i,'Pclass']
    s = train.loc[i,'Sex']
    train.loc[i,'Age'] = df2.loc[(e,p,s),'Age']
    
for i in train.index:
    age = train.loc[i,'Age']
    if age < 20:
        train.loc[i,'AgeGroup'] = '0~19'
    elif (age >= 20) & (age < 40):
        train.loc[i,'AgeGroup'] = '20~39'
    elif (age >= 40) & (age < 80):
        train.loc[i,'AgeGroup'] = '40~79'
    elif age >= 80:
        train.loc[i,'AgeGroup'] = '80~'
    else:
        continue

print()
print('정제 후 결측값')
print(train.isna().sum())
idx = train['Embarked'].isna()
train.loc[idx,:]

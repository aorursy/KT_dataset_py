# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')
train=data.copy()
index=train[train['winPlacePerc'].isnull()==True].index

train.drop(index,inplace=True)
train.info()
train.drop(columns={'killPlace','killPoints','killStreaks','maxPlace','winPoints'},inplace=True)
import matplotlib.pyplot as plt

import seaborn as sns
plt.style.use('fivethirtyeight')
f,size=plt.subplots(figsize=(15,10))
sns.heatmap(train.corr(),annot=True,fmt='.1f',ax=size)
walk=train.copy()
walk=walk[walk['walkDistance']<walk['walkDistance'].quantile(0.99)]
bins_new=plt.hist(walk['walkDistance'],bins=10)[1]

walk['walkDistance'] = pd.cut(walk['walkDistance'], bins_new, labels=['0-400m','400-850m', '850-1320m', '1320-1750m','1750-2190m','2190-2650m','2650-3080m','3080-3500m','3500-4000m','4000+'])
plt.figure(figsize=(15,8))
sns.boxplot(x="walkDistance", y="winPlacePerc", data=walk)

groups=train.copy()
groups=groups[groups['winPlacePerc']<=1 & (groups['winPlacePerc']>0.8)]
plt.figure(figsize=(15,8))
sns.countplot(y=groups['matchType'],data=groups)
comparison=train.copy()
comparison=comparison[comparison['winPlacePerc']<=1 & (comparison['winPlacePerc']>0.8)]
trace1=comparison['heals']<comparison['heals'].quantile(0.99)
trace2=comparison['boosts']<comparison['boosts'].quantile(0.99)

comparison=comparison[trace1 & trace2]
plt.figure(figsize=(15,8))
sns.distplot(comparison['heals'],hist=False,color='lime')
sns.distplot(comparison['boosts'],hist=False,color='blue')

plt.text(4,0.6,'heals',color='lime',fontsize = 17)
plt.text(4,0.55,'Boosts',color='blue',fontsize = 17)
plt.title('heals vs boosts')

vehicles=train.copy()
plt.figure(figsize=(15,8))
sns.pointplot(x='vehicleDestroys',y='winPlacePerc',data=vehicles)
plt.xlabel('Number of Vehicle Destroys')
plt.ylabel('Win Percentage')
plt.title('Vehicle Destroys/ Win Ratio')
plt.grid()
plt.show()
swim = train.copy()

swim['swimDistance'] = pd.cut(swim['swimDistance'], [-1, 0, 5, 20, 5286], labels=['0m','1-5m', '6-20m', '20m+'])

plt.figure(figsize=(15,8))
sns.boxplot(x="swimDistance", y="winPlacePerc", data=swim)



killings=train.copy()
plt.figure(figsize=(15,8))
sns.pointplot(x='kills',y='winPlacePerc',data=killings)
plt.xlabel('kills')
plt.ylabel('Win Percentage')
plt.title('kills/ Win Ratio')
plt.grid()
plt.show()
newdata=train.copy()
newdata=newdata[newdata['winPlacePerc']==1]
plt.figure(figsize=(15,8))
sns.pointplot(x='kills',y='winPlacePerc',data=killings,color='#CC0000')
sns.pointplot(x='headshotKills',y='winPlacePerc',data=killings,color='blue')

train['totalDistance'] = train['walkDistance']+train['rideDistance']+train['swimDistance']
train['boostsPerWalkDistance'] = train['boosts']/(train['walkDistance']+1)
df=train.copy()
df=df[df['boostsPerWalkDistance']<df['boostsPerWalkDistance'].quantile(0.99)]
bi=plt.hist(df['boostsPerWalkDistance'])[1]
bi

df['boostsPerWalkDistance'] = pd.cut(df['boostsPerWalkDistance'], bi, labels=['0-0.0008m','0.0009-0.0016m', '0.0017-0.0025m', '0.00266-0.0033m','0.0034-0.0041m','0.0042-0.0049m','0.005-0.0058m','0.0059-0.0065m','0.0066-0.0074m','0.0074+'])
plt.figure(figsize=(15,8))
sns.boxplot(y="boostsPerWalkDistance", x="winPlacePerc", data=df)
train['weaponsAcquiredPerWalkDistance'] = train['weaponsAcquired']/(train['walkDistance']+1)
train.columns
weapons=train.copy()
weapons=weapons[weapons['weaponsAcquiredPerWalkDistance']<weapons['weaponsAcquiredPerWalkDistance'].quantile(0.99)]
bi_=plt.hist(weapons['weaponsAcquiredPerWalkDistance'])[1]
bi_
weapons['weaponsAcquiredPerWalkDistance'] = pd.cut(weapons['weaponsAcquiredPerWalkDistance'], bi_, labels=['0-0.015m','0.016-0.031m', '0.032-0.046m', '0.047-0.062m','0.063-0.077m','0.078-0.093m','0.094-0.108m','0.109-0.124m','0.125-0.139m','0.139+'])
plt.figure(figsize=(15,8))
sns.boxplot(y="weaponsAcquiredPerWalkDistance", x="winPlacePerc", data=weapons)
train['team'] = ['1' if i>50 else '2' if (i>25 & i<=50) else '4' for i in train['numGroups']]

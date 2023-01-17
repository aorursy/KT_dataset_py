# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime
import seaborn as sns 
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')
df=data.copy()
df.shape
df.head()
# Expand the view so we have more visibility to the columns.
pd.set_option('display.max_columns',50)
df.head()
df.describe()
# taking one column from data and display all the unique variables
df['kills'].unique()
# method to change the format of the columns at the pandas level(2 decimal places)
pd.set_option('display.float_format', '{:.2f}'.format)
df.describe()
df.dtypes
# Check a single match
df[df["matchId"]=='4c8ae4cfd86290'].sort_values("winPlacePerc",ascending=0)
df=df.astype({'matchType':'category'})

#crosstabing data
ctab = pd.crosstab(df["kills"],df["winPlacePerc"])
ctab
sns.jointplot(x="winPlacePerc", y="kills", data=df, height=10, color="r")
plt.show()
print("The average person walks for {:.1f}m, 99% of people have walked {}m or less,The champions walked for {}m.".format(df['walkDistance'].mean(), df['walkDistance'].quantile(0.99), df['walkDistance'].max()))
print("{} players ({:.4f}%) walked 0 meters. This means that they die before even taking a step or dies as soon as landing (more possible).".format(len(df[df['walkDistance'] == 0]), 100*len(df[df['walkDistance']==0])/len(df)))
sns.jointplot(x="winPlacePerc", y="walkDistance", data=df, height=10, color="g")
plt.show()
print("The average person drives for {:.1f}m, 99% of people have drived {}m or less, while the formula 1 champion drived for {}m.".format(df['rideDistance'].mean(), df['rideDistance'].quantile(0.99), df['rideDistance'].max()))
sns.jointplot(x="winPlacePerc", y="rideDistance", data=df, height=10,  color="blue")

plt.show()
print("The average person kills {:.4f} players, 99% of people have {} kills or less, while the most kills ever recorded is {}.".format(df['kills'].mean(),df['kills'].quantile(0.99), df['kills'].max()))
plt.show()
df = df[df['kills']==0]
plt.figure(figsize=(15,10))
plt.title("Damage Dealt by 0 killers",fontsize=15)
sns.distplot(data['damageDealt'])

plt.show()
df['kills'][df['winPlacePerc']].plot(kind='hist')
# with zero kill
df['winPlacePerc'][df['kills']==0].hist()
# with one kill
df['winPlacePerc'][df['kills']==1].hist()
print("{} players ({:.4f}%) have won without a single kill!".format(len(df[df['winPlacePerc']==1]), 100*len(df[df['winPlacePerc']==1])/len(data)))

df1 = data[data['damageDealt'] == 0].copy()
print("{} players ({:.4f}%) have won without dealing damage!".format(len(df1[df1['winPlacePerc']==1]), 100*len(df1[df1['winPlacePerc']==1])/len(data)))
f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='vehicleDestroys',y='winPlacePerc',df=df,color='#606060',alpha=0.8)
plt.xlabel('Number of Vehicle Destroys',fontsize = 15,color='blue')
plt.ylabel('Win Percentage',fontsize = 15,color='blue')
plt.title('Vehicle Destroys/ Win Ratio',fontsize = 20,color='blue')
plt.grid()
plt.show()

print("The average person swims for {:.1f}m, 99% of people have swimemd {}m or less, while the olympic champion swimmed for {}m.".format(data['swimDistance'].mean(), data['swimDistance'].quantile(0.99), data['swimDistance'].max()))

#Code

df = df[df['heals'] < df['heals'].quantile(0.99)]
df = df[df['boosts'] < df['boosts'].quantile(0.99)]

f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='heals',y='winPlacePerc',df=df,color='lime',alpha=0.8)
sns.pointplot(x='boosts',y='winPlacePerc',df=df,color='blue',alpha=0.8)
plt.text(4,0.6,'Heals',color='lime',fontsize = 17,style = 'italic')
plt.text(4,0.55,'Boosts',color='blue',fontsize = 17,style = 'italic')
plt.xlabel('Number of heal/boost items',fontsize = 15,color='blue')
plt.ylabel('Win Percentage',fontsize = 15,color='blue')
plt.title('Heals vs Boosts',fontsize = 20,color='blue')

#Test
plt.grid()
plt.show()
sns.jointplot(x="winPlacePerc", y="heals", data=df, height=10, ratio=3, color="lime")
plt.show()
sns.jointplot(x="winPlacePerc", y="boosts", data=df, height=10, ratio=3, color="blue")
plt.show()
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
#Heatmap for the most positive variables

k = 5 #number of variables for heatmap
f,ax = plt.subplots(figsize=(11, 11))
cols = df.corr().nlargest(k, 'winPlacePerc')['winPlacePerc'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
sns.set()
cols = ['winPlacePerc', 'walkDistance', 'boosts', 'weaponsAcquired', 'damageDealt', 'killPlace']
sns.pairplot(df[cols], size = 2.5)
plt.show()

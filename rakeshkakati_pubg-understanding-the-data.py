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
# Importing the required libraries for this project

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

%matplotlib inline

import seaborn as sns

import itertools

import warnings 

warnings.filterwarnings("ignore")
 !wget https://www.dropbox.com/s/gax4zj8bfesaem8/PUBG.csv?dl=0
pubg = pd.read_csv('PUBG.csv?dl=0') 

pubg.head()
print("The average person kills {:.2f} players, 80% of people have {} kills or less, while the most kills ever recorded is {}.".format(pubg['kills'].mean(),pubg['kills'].quantile(0.80), pubg['kills'].max()))
pubg1 = pubg.copy()

pubg1.loc[pubg1['kills'] > pubg1['kills'].quantile(0.99)] = '8+' 

plt.figure(figsize=(10,5)) 

sns.countplot(pubg1['kills'].astype('str').sort_values()) 

plt.title("Kill Count",fontsize=25)

plt.show()
kills = pubg.copy()

kills['killsCategories'] = pd.cut(kills['kills'], [-1, 0, 2, 5, 10, 60], labels=['0_kills','2-4_kills', '4-5_kills', '7-9_kills', '10+_kills'])

plt.figure(figsize=(14,6))

sns.boxplot(x="killsCategories", y="winPlacePerc", data=kills)

plt.show()
print("The average person walks for {:.2f}m, 80% of people have walked {}m or less, while the marathoner champion walked for {}m.".format(pubg['walkDistance'].mean(), pubg['walkDistance'].quantile(0.80), pubg['walkDistance'].max()))

pubg2 = pubg.copy()

pubg2 = pubg2[pubg2['walkDistance'] < pubg['walkDistance'].quantile(0.99)] 

plt.figure(figsize=(20,15))

plt.title("Walking Distance Distribution",fontsize=20) 

sns.distplot(pubg2['walkDistance'])

plt.show()

sns.jointplot(x="winPlacePerc", y="walkDistance", data=pubg, height=10, ratio=3, color ="green")

plt.show()
f,ax1 = plt.subplots(figsize =(20,10)) 

sns.pointplot(x='vehicleDestroys',y='winPlacePerc',data=pubg2,color='red',alpha=0.8) 

plt.xlabel('Number of Vehicle Destroys',fontsize = 15,color='blue') 

plt.ylabel('Win Percentage',fontsize = 15,color='blue')

plt.title('Vehicle Destroys/ Win Ratio',fontsize = 20,color='blue')

plt.grid()

plt.show()
print("The average person uses {:.2f} heal items, 90% of people use {} or less".format( pubg['heals'].mean(), pubg['heals'].quantile(0.90), pubg['heals'].max()))

print("The average person uses {:.2f} boost items, 90% of people use {} or less".format (pubg['boosts'].mean(), pubg['boosts'].quantile(0.90), pubg['boosts'].max()))
pubg3 = pubg.copy()

pubg3 = pubg3[pubg3['heals'] < pubg3['heals'].quantile(0.90)] 

pubg3 = pubg3[pubg3['boosts'] < pubg3['boosts'].quantile(0.90)]

f,ax1 = plt.subplots(figsize =(20,10)) 

sns.pointplot(x='heals',y='winPlacePerc',data=pubg3,color='green',alpha=0.8) 

sns.pointplot(x='boosts',y='winPlacePerc',data=pubg3,color='blue',alpha=0.8) 

plt.text(4,0.6,'Heals',color='green',fontsize = 20,style = 'normal') 

plt.text(4,0.55,'Boosts',color='blue',fontsize = 20,style = 'normal') 

plt.xlabel('Number of heal/boost items',fontsize = 15,color='black') 

plt.ylabel('Win Percentage',fontsize = 15,color='black')

plt.title('Heals vs Boosts',fontsize = 20,color='black') 

plt.grid()

plt.show()
solo = pubg[pubg['numGroups']>50]

duo = pubg[(pubg['numGroups']>25) & (pubg['numGroups']<=50)]

squads = pubg[pubg['numGroups']<=25]

print("There are {} ({:.3f}%) solo games, {} ({:.3f}%) duo games and {} ({:.3f}%) squad games.".format(len(solo), 100*len(solo)/len(pubg), len(duo), 100*len(duo)/len(pubg), len(squads), 100*len(squads)/len(pubg),))

f,ax1 = plt.subplots(figsize =(20,10)) 

sns.pointplot(x='kills',y='winPlacePerc',data=solo,color='black',alpha=0.8) 

sns.pointplot(x='kills',y='winPlacePerc',data=duo,color='green',alpha=0.8) 

sns.pointplot(x='kills',y='winPlacePerc',data=squads,color='brown',alpha=0.8) 

plt.text(37,0.6,'Solos',color='black',fontsize = 15,style = 'normal') 

plt.text(37,0.55,'Duos',color='green',fontsize = 15,style = 'normal') 

plt.text(37,0.5,'Squads',color='brown',fontsize = 15,style = 'normal') 

plt.xlabel('Number of kills',fontsize = 20,color='orange')

plt.ylabel('Win Percentage',fontsize = 20,color='orange') 

plt.title('Solo vs Duo vs Squad Kills',fontsize = 20,color='orange') 

plt.grid()

plt.show()
f,ax = plt.subplots(figsize=(25, 25)) 

sns.heatmap(pubg.corr(), annot=True, linewidths=.5, ax=ax) 

plt.show()
# Now we shall plot the heatmap for the features mentioned above

k = 7 #number of variables for heatmap

f,ax = plt.subplots(figsize=(11, 11))

cols = pubg.corr().nlargest(k, 'winPlacePerc')['winPlacePerc'].index 

cm = np.corrcoef(pubg[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
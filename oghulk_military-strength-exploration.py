import numpy as np

import pandas as pd



df = pd.read_csv('../input/GlobalFirePower.csv')



df.head()
# Here I checked the data to see what might need to be done to it, and surprisingly I found that

# it was mostly already very clean and not needing any work done on it!

df.info()
import seaborn as sns

import matplotlib.pyplot as plt



topTen = df.loc[df.Rank < 10]

sns.barplot(x='ISO3',y='Manpower Available',data=topTen)
sns.barplot(x='Total Military Personnel',y='ISO3',data=topTen,

           label='Total Military Personnel',color='#d2894c')

sns.barplot(x='Active Personnel',y='ISO3',data=topTen,

            label='Active-Duty Personnel', color='#BF5700')
sns.barplot(x='Total Aircraft Strength',y='ISO3',data=topTen,color='#d2894c')

sns.barplot(x='Fighter Aircraft',y='ISO3',data=topTen,color='#bf5700')
sns.barplot(x='Total Aircraft Strength',y='ISO3',data=topTen,color='#d2894c')

sns.barplot(x='Attack Aircraft',y='ISO3',data=topTen,color='#bf5700')
sns.barplot(x='ISO3',y='Aircraft Carriers',data=topTen)
f, axarr = plt.subplots(3, 2, figsize=(10,10))

sns.barplot(x='ISO3',y='Frigates',data=topTen,ax=axarr[0,0])

sns.barplot(x='ISO3',y='Destroyers',data=topTen,ax=axarr[0,1])

sns.barplot(x='ISO3',y='Corvettes',data=topTen,ax=axarr[1,0])

sns.barplot(x='ISO3',y='Submarines',data=topTen,ax=axarr[1,1])

sns.barplot(x='ISO3',y='Patrol Craft',data=topTen,ax=axarr[2,0])

sns.barplot(x='ISO3',y='Mine Warfare Vessels',data=topTen,ax=axarr[2,1])
f, axarr = plt.subplots(3, 2, figsize=(10,10))

sns.barplot(x='ISO3',y='Attack Helicopters',data=topTen,ax=axarr[0,0])

sns.barplot(x='ISO3',y='Combat Tanks',data=topTen,ax=axarr[0,1])

sns.barplot(x='ISO3',y='Armored Fighting Vehicles',data=topTen,ax=axarr[1,0])

sns.barplot(x='ISO3',y='Self-Propelled Artillery',data=topTen,ax=axarr[1,1])

sns.barplot(x='ISO3',y='Towed Artillery',data=topTen,ax=axarr[2,0])

sns.barplot(x='ISO3',y='Rocket Projectors',data=topTen,ax=axarr[2,1])
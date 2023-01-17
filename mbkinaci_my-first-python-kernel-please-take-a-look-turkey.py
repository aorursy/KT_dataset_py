#Hello my friends , I passed from R to Python one month ago . This is my first Python kernel. If you

# take a look and make a comment , I would appreciate you.

#Importing libraries

import pandas as pd

import numpy as np

import scipy as sci

import seaborn as sns

import matplotlib.pyplot as plt





dataset = pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding='ISO-8859-1')



dataset.shape
dataset = dataset.loc[:,['eventid','iyear','imonth','iday','country_txt','city','latitude','longitude',

                    'attacktype1_txt','targtype1_txt','gname','weaptype1_txt',

                    'nkill','nwound']]
#filtering Turkey

turkey = dataset[dataset.country_txt == 'Turkey']
turkey.shape
#missing value checking

import missingno as msno

msno.bar(turkey,sort=True)
#Data preprocessing
turkey = turkey.rename(columns = {'eventid':'id','imonth':'month','iyear':'year','iday':'day','country_txt':'country',

                                  'attacktype1_txt':'attack_type','targtype1_txt':'target_type',

                                  'weaptype1_txt':'weapon_type','nkill':'no_of_victims','nwound':'no_of_wounds'})
turkey['no_of_victims']=turkey.no_of_victims.replace(np.nan,0)

turkey['no_of_wounds']=turkey.no_of_wounds.replace(np.nan,0)
turkey['city'] =  turkey['city'].apply(lambda x: x.lower())

##Target_type vs Year

a=turkey.groupby(['target_type','year']).size()

a

b=a.unstack(level=0)



e=b.replace(np.nan,0)

e=e.astype(int)



import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(15, 12)) 

gs = gridspec.GridSpec(1, 2, width_ratios=[10, 1.5]) 

ax0 = plt.subplot(gs[0])

sns.heatmap(e,annot=True,fmt='d',linewidths=.5,ax=ax0, cbar=False,annot_kws={"size":10})

ax1 = plt.subplot(gs[1],sharey=ax0)

sns.heatmap(pd.DataFrame(e.sum(axis=1)),annot=True,fmt='d',linewidths=.5,ax=ax1,cbar=False,annot_kws={"size":10})

plt.setp(ax1.get_yticklabels(), visible=False)

plt.setp(ax1.set_ylabel([]),visible=False)

plt.setp(ax0.yaxis.get_majorticklabels(),rotation=0)

ax0.tick_params(axis='y',labelsize=16)

ax0.tick_params(axis='x',labelsize=16)

ax0.set_ylabel("Year",size=18)

ax0.set_xlabel("Target Type",size=18)

ax1.set_xticklabels(["Total"],size=16)

ax0.set_title("Target Type vs Year ",size=22,y=1.05,x=0.5)



plt.show()

#Mostly Military and Police members were attacked
a=turkey.groupby(['weapon_type','year']).size()

a

b=a.unstack(level=0)



e=b.replace(np.nan,0)

e=e.astype(int)



import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(15, 12)) 

gs = gridspec.GridSpec(1, 2, width_ratios=[10, 1.5]) 

ax0 = plt.subplot(gs[0])

sns.heatmap(e,annot=True,fmt='d',linewidths=.5,ax=ax0, cbar=False,annot_kws={"size":10})

ax1 = plt.subplot(gs[1],sharey=ax0)

sns.heatmap(pd.DataFrame(e.sum(axis=1)),annot=True,fmt='d',linewidths=.5,ax=ax1,cbar=False,annot_kws={"size":10})

plt.setp(ax1.get_yticklabels(), visible=False)

plt.setp(ax1.set_ylabel([]),visible=False)

plt.setp(ax0.yaxis.get_majorticklabels(),rotation=0)

ax0.tick_params(axis='y',labelsize=16)

ax0.tick_params(axis='x',labelsize=16)

ax0.set_ylabel("Year",size=18)

ax0.set_xlabel("Weapon Type",size=18)

ax1.set_xticklabels(["Total"],size=16)

ax0.set_title("Weapon Type vs Year ",size=22,y=1.05,x=0.5)



plt.show()
#Explosives/Bombs/Dynamite and Firearms are most outstanding weapon types.
##Cities affected most

turkey = turkey[turkey['city'] != 'unknown']

klm = turkey.groupby('city')['id'].count().sort_values(ascending=False).head(20)

klm = pd.DataFrame(klm)

klm['city']=klm.index



sns.factorplot(x='city', 

               y="id",data=klm, 

               saturation=1, 

               kind="bar", 

               ci=None, 

               aspect=1, 

               linewidth=1) 

locs, labels = plt.xticks()

plt.setp(labels, rotation=90)
#Istanbul, Ankara,Diyarbakir and Izmir are affected by terror attacks most. 
#Month

sns.despine()

sns.distplot(turkey['month'],kde_kws={"color":"g","lw":4,"label":"KDE Estimation","alpha":0.5},

            hist_kws={"color":"r","alpha":0.5,"label":"Frequency"});

plt.xlim(0,13)

plt.xticks(np.arange(1,12),size=12)

plt.yticks(size=14)

plt.ylabel("Density",rotation=90, size=20)

plt.xlabel("Month",size=20)

plt.show()
#July and August are most dangerous times to visit Turkey.
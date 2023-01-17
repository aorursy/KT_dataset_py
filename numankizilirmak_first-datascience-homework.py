# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Get data from csv

terrorData=pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')
#rename columns

terrorData.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)

#convert to dataframe

terrorData=terrorData[['Year','Month','Day','Country','Region','city','latitude','longitude','AttackType','Killed','Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]



terrorData['HarmToPeopleCount']=terrorData['Killed']+terrorData['Wounded']



#get 5 rows of data for view

terrorData.head(5)
terrorData.corr()
deadlyAttacks=terrorData[terrorData['Killed']>0]

print('Min killed in an attack :',deadlyAttacks['Killed'].min(),'in',deadlyAttacks.loc[deadlyAttacks['Killed'].idxmin()].Country)
print('Max killed in an attack :',terrorData['Killed'].max(),'in',terrorData.loc[terrorData['Killed'].idxmax()].Country)
plt.subplots(figsize=(15,5))

sns.countplot('Year',data=terrorData,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',9))

plt.xticks(rotation=90)

plt.title('Number Of Attacks  Each Year')

plt.show()

plt.subplots(figsize=(15,5))

sns.countplot('Year',data=deadlyAttacks,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',9))

plt.xticks(rotation=90)

plt.title('Number Of Attacks  Each Year(Caused Death)')

plt.show()
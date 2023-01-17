# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
fname = os.listdir("../input")[0]
data = pd.read_csv('../input/'+fname)
data.head()
# Any results you write to the current directory are saved as output.
data.describe()
import seaborn as sns
#import collections

#log_kill = dict(collections.Counter(data.n_killed.values))
import matplotlib.pyplot as plt
#sns.jointplot(np.array(log_kill.keys()),np.log(np.array(list(log_kill.values()))),kind='hex')
ax=plt.subplot(2,1,1)
ax.hist(data.n_injured,width=0.8,log=True,bins = 50,color='b')
plt.title('People Injured Dist.')
ax2=plt.subplot(2,1,2)
plt.title('People Dead Dist.')
ax2.hist(data.n_killed,width=0.8,log=True,bins=50,color='r')
print(data.n_injured.describe())
print(data.n_killed.describe())
#ax.set_yscale('log')

region=data.set_index(['state','city_or_county'])
region.head()
ax=plt.subplots(figsize=(20,20))
#ax = plt.subplot(1,1,1)
plt.xticks(rotation=90)
plt.title('Gun Violence in Different States')
sns.countplot(data.sort_values('state').state)

#print(region.index.value_counts())
print('GUN VIOLENCE: STATE AND CITY/COUNTY Wise: ')
print(region.index.value_counts())
region.index.value_counts().plot()
dated = pd.DataFrame(data[['n_killed','n_injured']].values , index = data.date, columns = ['Killed','Injured'])
dated = dated.groupby(dated.index).agg({'Killed':sum , 'Injured': sum})
dated.head()
dated.dropna()
dated.sort_index(inplace=True)
#ax = plt.subplot(1,1,1)
#plt.xlabel(dated.index.values)
ax=dated.plot(figsize=(20,20),rot=90,title='Gun Violence with progressing time')
#plt.xticks(np.arange(dated.index.values.shape[0]))
print(dated.describe())
#print(type(dated.index.values[0]))
import re
guns=data[['n_guns_involved','gun_type']]
guns.dropna(inplace= True)
guns.describe()
ax = plt.subplots(figsize=(20,20))
ax1= plt.subplot(2,2,1)
ax1.hist(guns['n_guns_involved'], bins = 40 , width = 9.0 ,log=True)
plt.title('Guns Involved in Numbers')
types=[]

for i in guns.iterrows():
   # types.append('a')
    types.extend(list(re.findall(r'(\.?\d*[a-zA-Z]+[-\w]*)',i[1]['gun_type'])))

import collections
#print(len(types))
#print(list(re.findall(r'([a-zA-Z]+)',i[1]['gun_type'])))
coll=dict(collections.Counter(types))
print(coll)
ax2 = plt.subplot(2,2,2)
ax2.bar(*zip(*coll.items()))
#ax2.xticks()
plt.title('Types of guns')
plt.xticks(rotation='vertical')
#ax2.xaxis?
#ax2.bar(types)

gender = data[['participant_status','participant_gender']]
genders=[]
stati=[]
gender.dropna(how='any',inplace=True)
for i in gender.iterrows():
    #1+1
    try:
        g1=list(re.findall(r'([a-zA-Z]+)',i[1]['participant_gender']))
        s1=list(re.findall(r'([a-zA-Z]+)',i[1]['participant_status']))
        if len(g1)==len(s1):
            genders.extend(g1)
            stati.extend(s1)
        #else:
            #print(i[1])
    except:
        print(i[1])
#print(                 #     list(re.findall(   r'([a-zA-Z]+)' ,i[1]['participant_status']))       )
print([len(genders),len(stati)])
gender = pd.DataFrame(list(zip(genders,stati)),columns=['Gender','Status'])
gender.head()
'''
ax1=plt.subplot(2,2,1)
plt.title('Killed')
ax1.bar(gender[gender['Status']=='Killed'].Gender)
ax2=plt.subplot(2,2,2)
ax2.bar(gender[gender['Status']=='Injured'].Gender)
'''
import collections
print('Killed')
print(collections.Counter(list(gender[gender['Status']=='Killed'].Gender.values)))
print('Injured')
print(collections.Counter(list(gender[gender['Status']=='Injured'].Gender.values)))
print('Unharmed')
print(collections.Counter(list(gender[gender['Status']=='Unharmed'].Gender.values)))
print('Arrested')
print(collections.Counter(list(gender[gender['Status']=='Arrested'].Gender.values)))
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
try:
    file = pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')
    print('File load: Success')
except:
    print('File load: Failed')
print('Country with 1. Highest Terrorist Attacks :', 
      file["country_txt"].value_counts().index[0], 
      ', Number of Attacks :', file["country_txt"].value_counts()[0])
print('Country with 2. Highest Terrorist Attacks :', 
      file["country_txt"].value_counts().index[1], 
      ', Number of Attacks :', file["country_txt"].value_counts()[1])
print('Country with 3. Highest Terrorist Attacks :', 
      file["country_txt"].value_counts().index[3], 
      ', Number of Attacks:', file["country_txt"].value_counts()[3])
print(".")
print(".")
print('Country with 3. Lowest Terrorist Attacks :', 
      file["country_txt"].value_counts().index[-3], 
      ', Number of Attacks:', file["country_txt"].value_counts()[-3])
print('Country with 2. Lowest Terrorist Attacks :', 
      file["country_txt"].value_counts().index[-2], 
      ', Number of Attacks:', file["country_txt"].value_counts()[-2])
print('Country with 1. Lowest Terrorist Attacks :', 
      file["country_txt"].value_counts().index[-1], 
      ', Number of Attacks:', file["country_txt"].value_counts()[-1])
# Sorted from highest to lowest Terrorist Attacks

print(file["country_txt"].value_counts(normalize=False, sort=True, ascending=False, bins=None, dropna=True))


# Filtering country name

file_turkey = file[file.country_txt == 'Turkey']

print('City with Highest Terrorist Attacks in Turkey:', 
      file_turkey['city'].value_counts().index[0], ", number of attacks : ", file_turkey['city'].value_counts()[0])
print('Group with Highest Terrorist Attacks in Turkey:', 
      file_turkey['gname'].value_counts().index[0], ", number of attacks : ", file_turkey['gname'].value_counts()[0])
print('Maximum People Killed in a Terrorist Attack in Turkey:', 
      file_turkey['nkill'].sort_values(ascending = False).iloc[0], 'Group:', file_turkey.loc[file_turkey['nkill'].idxmax(),'gname'])
liste_country= []
liste_count=[]
liste_country_count=[]
for i in file["country_txt"].unique():
    liste_country.append(i)
for i in file["country_txt"]:
    liste_count.append(i)

for i in liste_country:
    b= liste_count.count(i)
    liste_country_count.append(b)

dictionary = dict(zip(liste_country, liste_country_count))
print (sorted([(value,key) for (key,value) in dictionary.items()],reverse=True ))


key_max = max(dictionary.keys(), key=(lambda k: dictionary[k]))
key_min = min(dictionary.keys(), key=(lambda k: dictionary[k]))

print('Maximum Value: ',dictionary[key_max])
print('Minimum Value: ',dictionary[key_min])

file_turkey.iyear.plot(kind = 'hist', grid = True, color = 'r',bins = range(1970,2018), figsize = (14,6),alpha = 0.6)
plt.xticks(range(1970,2018),rotation = 45, fontsize = 10)
plt.yticks(range(0,750,25), fontsize = 10)
plt.xlabel('Year', fontsize = 12)
plt.ylabel('Number of Terrorist Attacks',fontsize = 12)
plt.title('Number of Terrorist Attacks By Year', fontsize = 14)
plt.xlim((1970,2018))
plt.show()

# How many terrorisim attacks happened in each country
plt.subplots(figsize=(100,15))
sns.countplot(file['country_txt'], data=file)
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities for Each Country')
plt.show()
# There are greater than 10 attack from Province of Turkey
x=file_turkey.provstate.value_counts()>100
print(file_turkey.provstate.value_counts()[x])
file_turkey.provstate.value_counts().drop('Unknown').head(15).plot(kind = 'bar', figsize = (14,6), grid = True )
plt.ylabel('Number of Terrorist Attacks',fontsize = 12)
plt.xlabel("Province",fontsize = 12 , rotation =45)
plt.title('Most Targeted Province ', fontsize = 14)

plt.show()

file_turkey.provstate.value_counts().drop('Unknown').head(20).plot(kind = 'pie', figsize = (10,10) )
plt.title('Most Targeted Province', fontsize = 14)

plt.show()
file_turkey.weaptype1_txt.value_counts().drop('Unknown').head(4).plot(kind = 'pie', figsize = (10,10) )
plt.title('Prefer Weapons', fontsize = 14)

plt.show()
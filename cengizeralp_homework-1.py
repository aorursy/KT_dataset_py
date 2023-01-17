# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1') # w/o encoding, data can't be downloaded
data.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType',
                     'target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type',
                     'weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)  
# Column names are changed (taken from Ashwini Swain - Terrorism Around The World - https://www.kaggle.com/ash316/terrorism-around-the-world)
data.info()
data.columns  # there are too many columns 
data = data[['Year','Month','Day','Country','Region','city','latitude','longitude','AttackType','Killed','Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]
data['casualities']=data['Killed']+data['Wounded']
# Column numbers are reduced (taken from Ashwini Swain - Terrorism Around The World - https://www.kaggle.com/ash316/terrorism-around-the-world)
data.columns
data.head()
data.corr()
#correlation map
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(data.corr(), annot=True, linewidths=0.5, fmt= '.1f',ax=ax) # 
plt.show()
# line plot
plt.subplots(figsize=(20,7))
data.Wounded.plot (kind ='line', color='g', label = 'Wounded', linestyle =':', grid=True, alpha = 0.5, linewidth = 2 )
data.Killed.plot (kind ='line', color='r', label = 'Killed', linestyle ='-.', grid=True, alpha = 0.5, linewidth = 2 )
plt.legend()
plt.xlabel ('index')
plt.ylabel ('number')
plt.title ('Line Plot')
plt.show()
# Scatter Plot
# x = Wounded, y = Killed (x must be an input feture. But I choose 2 outputs to see the relation between them, )
# plt.subplots(figsize=(10,7))  # problem seen - cancelled
data.plot ( kind ='scatter', x= 'Wounded', y= 'Killed', alpha = 0.8, color ='purple')
plt.title('Scatter Plot - Wounded vs. Killed')
plt.show()
# Histogram 1
data.Year.plot( kind = 'hist', bins = 50, figsize=(20,7) )
# data.Wounded.plot( kind = 'hist' )
plt.show()
# Histogram 2  
data.latitude.plot( kind = 'hist', bins = 12, figsize=(20,7) )
plt.show()
series = data['Killed']
data_frame = data[['Killed']]
print(type(series))
print(type(data_frame))
print(3<2)
print(3>2)
print(3!=2)
print(True and False)
print(True or False)
x = data['Killed'] >1000
data[x]
data[(data['Killed']>1000) & (data ['Wounded']>1000)]
data[np.logical_and(data['Killed']>1000, data['Wounded']>1000)]
# Stay in loop if condition (i is different from 5) is true
i = 0
while i != 5 :
    print('i is equal to : ', i)
    i += 1
print (i, 'is equal to 5')
lis = {1,2,3,4,5}
for i in lis:
    print('i is : ', i)
print('')

for index, value in enumerate(lis):
    print(index, ' : ', value)
print('')

dictionary = {'spain':'madrid','france':'paris'}
for key, value in dictionary.items():
    print(key,' : ', value)

for index, value in data[['Killed']][0:5].iterrows():
    print(index,' : ',value)

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/fifa-international-soccer-mens-ranking-1993now/fifa_ranking.csv')
data.info()

data.head(10)
data.tail()
data.corr()
f,ax = plt.subplots(figsize=(20, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.columns


dataFrame1=pd.DataFrame(data)
dataFrame1.describe()#dataframe  yapılarak std özelliği kontrol edildi
dataFrame1.dtypes



dataFrame1.loc[:100,["rank","country_full"]]
dataFrame1.loc[:,::-1]#ters çevirdik deneme amaçlı yapılmaktadır
filter1=dataFrame1.previous_points>=50
filter2=dataFrame1.total_points<18
dataFrame1[filter1&filter2] #filtreleme 
avgPdTotalPoints=dataFrame1.total_points.mean()
avgPdTotalPoints

dataFrame1["puan"]=["dusuk" if each<=avgPdTotalPoints else "yuksek"for each in dataFrame1.total_points]#ortalamanın altındakilere düşük
                      
dataFrame1

data1=dataFrame1.head()
data2=dataFrame1.tail()
data_concat=pd.concat([data1,data2],axis=0)
data_concat
dataFrame1.iloc[3:5,2:9]#3 ten  5. satıra kadar 2den 9. sutununa kadar

data.three_year_ago_avg.plot(kind = 'line', color = 'blue',label = 'three_year_ago_avg',linewidth=1,linestyle = ':',figsize = (12,12))
data.total_points.plot(color = 'r',label = 'total_points',linewidth=1, alpha = 0.5,linestyle = '-.',grid=True)
plt.legend("ab")     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot uygulması')            # title = title of plot
plt.show()
data.total_points.plot(kind = 'hist',bins=10,figsize = (12,12))
plt.show()
dictionary = {'ankara' : 'odtü','afyon' : 'akü','trabzon':'ktü'}
print(dictionary.keys())
print(dictionary.values())
dictionary['trabzon'] = "avr"    
print(dictionary)
dictionary['sakarya'] = "saü"        
print(dictionary)
del dictionary['afyon']             
print(dictionary)
print('sakarya' in dictionary)        
dictionary.clear()                  
print(dictionary)
series = data['total_points']        
print(type(series))
data_frame = data[['total_points']]
print(type(data_frame))

sonuc=np.logical_and(data['total_points']<100, data['three_year_ago_avg']>200 )
data[sonuc]

sonuc=np.logical_or(data['total_points']<100, data['three_year_ago_avg']>200 )
data[sonuc]
for index,value in data[['country_full']][0:4].iterrows():
    print(index," : ",value)

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
#Import Data
data=pd.read_csv("../input/Iris.csv")
data.info()

#Find corelation between data types.
data.corr()
#Which type of datas can be found in my data?
data.columns
#Write the first 10 row of data
data.head(10)
#Write theend of 10 row of data
data.tail(10)
#correlation map
f,ax = plt.subplots(figsize=(20, 20))
sns.heatmap(data.corr(), annot=True, linewidths=.15, fmt= '.4f',ax=ax)
plt.show()
data = data.drop(["Id"],axis=1)#axis=1 tüm sutunu sil demek,axis=0 satır sil.
#dataFrame1.drop(["yeni_feature"],axis=1,inplace = True)
#correlation map(without ID)
f,ax = plt.subplots(figsize=(20, 20))
sns.heatmap(data.corr(), annot=True, linewidths=.15, fmt= '.4f',ax=ax)
plt.show()
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.SepalWidthCm.plot(kind = 'line', color = 'blue',label = 'SepalWidthCm',linewidth=2,alpha = 0.8,grid = True,linestyle = ':')
data.PetalLengthCm.plot(color = 'red',label = 'PetalLengthCm',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='lower left')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
# Scatter Plot 
# x = PetalLengthCm, y = PetalWidthCm
data.plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm',alpha = 0.9,color = 'green')
plt.xlabel('PetalLengthCm')              # label = name of label
plt.ylabel('PetalWidthCm')
plt.title('PetalLengthCm PetalWidthCm Scatter Plot')    # title = title of plot
plt.show()
#Histogram Pilot
data.PetalLengthCm.plot(kind = 'hist',bins = 40,figsize = (12,12))
plt.show()
# clf() = cleans it up again you can start a fresh
data.PetalLengthCm.plot(kind = 'hist',bins = 50)
plt.clf()
#create dictionary and look its keys and values
dictionary = {'ankara' : '06','istanbul' : '34','izmir' : '35'}
#ankara=06
#istanbul=34
#izmir=35
#erzurum=25

print(dictionary)
print(dictionary.keys())
print(dictionary.values())
dictionary['ankara'] = "0606"    # update existing entry
print(dictionary)
dictionary['istanbul'] = "3434"       # Add new entry
print(dictionary)
#del dictionary['izmir']              # remove entry with key 'spain'
print(dictionary)
print('istanbul' in dictionary)        # check include or not
dictionary['erzurum'] = "25"           #Add new key and value
print(dictionary)
dictionary.clear()                   # remove all entries in dict
print(dictionary)
# 1 - Filtering Pandas data frame
x = data['SepalLengthCm']>7.4   
data[x]
# This is also same with previous code line. Therefore we can also use 'or' for filtering.
data[(data['SepalLengthCm']>7.7) | (data['PetalWidthCm']>2.2)]
# For pandas we can achieve index and value
for index,value in data[['SepalLengthCm']][0:5].iterrows():
    print(index," : ",value)

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
data = pd.read_csv("../input/athlete_events.csv")
data.info()
data.head()
data.corr()
# Correletaion map
f,ax=plt.subplots(figsize=(10,10))
sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt=".2f",ax=ax)
plt.show()
# Also we can see from here weight and height is directly proportional
# Learn what is there as properties in data.
data.columns
data.head()
# Line plot I took first 800 data because all data includes 271116 and this is very big to analyze
data.Height[0:800].plot(kind = 'line', color = 'g',label = 'Height',linewidth=1,alpha = 0.7,grid = True,linestyle = ':')
data.Weight[0:800].plot(color="red",figsize=(8,8),label="Weight",linewidth=1,alpha=0.7,grid=True,linestyle=":")
plt.legend()
plt.xlabel("Athletes")
plt.ylabel("Height and Weight")
plt.title("Line Plot")
plt.show()





# Scatter Plot
data.plot(kind='scatter', x="Height", y="Weight",alpha = 0.7,color = 'red')
plt.xlabel('Height')              # label = name of label
plt.ylabel('Weight')
plt.title('Height Weight Scatter Plot')   
plt.show()
print(" max height of athletes is ",data.Height.max()," cm")
print(" max weight of athletes is ",data.Weight.max()," kg")
print(" min height of athletes is ",data.Height.min()," cm")
print(" min weight of athletes is ",data.Weight.min()," kg")




# histogram 
data.Height.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
#these are just for learning and practice
dictionary = {'Turkey' : ['istanbul','ankara'],'Usa' : ['Las_vegas''New_york']}
print(dictionary.keys())
print(dictionary.values())
dictionary['Turkey'] = ["balıkesir","bursa"]    # update existing entry
print(dictionary)
dictionary['france'] = ["paris","lille"]       # Add new entry
print(dictionary)
del dictionary['france']              # remove entry with key 'spain'
print(dictionary)
print('Turkey' in dictionary)        # check include or not
dictionary.clear()                   # remove all entries in dict
print(dictionary)
series_name = data['Name']        # data['Defense'] = series
print(type(series_name))
data_frame = data[['Age']]  # data[['Defense']] = data frame
print(type(data_frame))
x = data['Height']>220    
data[x]
y=data["Weight"]>200
print("max height and weight")
data[y]
data[(data['Height']>200) & (data['Weight']>150)]
for index,value in data[['Height']][0:8].iterrows():
    print(index," : ",value)
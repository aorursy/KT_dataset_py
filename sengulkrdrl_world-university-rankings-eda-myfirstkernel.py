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

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/cwurData.csv')

data.info()
data.corr()
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)  #creating correlation map
plt.show()
data.head(15) #first15
data.tail(15) #last15
data.columns
data.corr()  #corr. table
#this is line plot
data.world_rank.plot(kind = 'line', color = 'g',label = 'world_rank',linewidth=2,alpha = 1,grid = True,linestyle = ':')
data.national_rank.plot(color = 'r',label = 'quality_of_faculty',linewidth=0.5, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     
plt.xlabel('x axis')              
plt.ylabel('y axis')
plt.title('Line Plot')            
plt.show()


#this is scatter plot
data.plot(kind='scatter', x='quality_of_education', y='patents',alpha = 1,color = 'blue')
plt.xlabel('Quality Of Education')            
plt.ylabel('Patents')
plt.title('Quality Of Education-Patents Scatter Plot')  
plt.show()
# this is histogram
data.influence.plot(kind = 'hist',bins = 30,figsize = (7.5,5),color='b')
plt.show()
data.world_rank.plot(kind = 'hist',bins = 30,figsize = (7.5,5), color = 'b')
plt.show()

x = data['national_rank'] > 220            # Filtering Pandas data frame
data[x]
data[np.logical_and(data['national_rank'] > 200, data['world_rank'] > 900)]  # Filtering pandas with logical_and

len(data[ (data['national_rank'] > 200 ) & (data['world_rank'] > 900)])  #other way for filtering and using with len

import builtins   #built in scope
dir(builtins)
avgr = sum(data.score)/ len(data.score)
data["score"] = [ "high" if  score > avgr else "low" for score in data.score]
data.loc[:10,["score","score"]]                                           #?


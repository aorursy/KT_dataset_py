# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns #visualization tool

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/world-happiness/2019.csv') # 
for index,value in data[['Country or region']][78:79].iterrows(): # Overall rank of Turkey , we can see only Turkey.
    print(index," : ",value)
data.head() # We can see first 5 lines.


data.columns # We can see columns' names.
data.info() # for information in general
data.corr() # we can see the correlation between the data.
data.head(10) #first 5 lines
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=0.5, fmt= '.2f',ax=ax) #correlation map
plt.show() # block the information line
data.Score.plot(kind = 'line', color = 'blue',label = 'Score',linewidth=1,grid = True,linestyle = ':')
data.Generosity.plot(color = 'red',label = 'Generosity',linewidth=1,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
# Scatter Plot 
# x = attack, y = defense
data.plot(kind='scatter', x='Score', y='Generosity',alpha = 0.5,color = 'red')
plt.xlabel('Score')              # label = name of label
plt.ylabel('Generosity')
plt.title('Score and Generosity Scatter Plot')            # title = title of plot
data.Score.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
# clf() = cleans it up again you can start a fresh
data.Score.plot(kind = 'hist',bins = 50)
plt.clf()
# We cannot see plot due to clf()
# Filtering pandas data frame
x = data['Score']>7   # There are only 16 countries that have higher score value than 7
data[x]
data[(data['Score']>7) & (data['Generosity']>0.2)] #There are only 13 countries that have higher score value than 7 and higher generosity value than 0.2 
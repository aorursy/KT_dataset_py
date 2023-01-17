# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/2015.csv")
data.info() # Information for data
data.describe() # Description of data
data.corr()
f,ax = plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(), annot=True,linewidth=5, fmt='.1f',ax=ax)
plt.show()

# print Columns of data with their number
for index,value in enumerate(data.columns):  
    print(index+1,value)
    
data.columns = [each.replace(" ","_") if (len(each.split()) > 1) else each for each in data.columns]
data.columns

data.Freedom.plot(kind="line",color='red',label='Freedom',grid=True, alpha=0.5,linewidth=1,linestyle=':')
data.Generosity.plot(kind='line',color='blue',label='Generosity',grid=True, alpha=0.5,linewidth=1,linestyle='-.')
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()


data.plot(kind='scatter', x='Freedom',y='Generosity',alpha=0.5, color='blue',grid=True)
plt.xlabel('Freedom')
plt.ylabel('Generosity')
plt.title('Scatter Plot')
plt.show()
data.Freedom.plot(kind='hist',bins=50,figsize=(12,8))
plt.show()
data.Freedom.plot(kind='hist',bins=50)
plt.clf()
print('Average Happiness Score: ',data.Happiness_Score.mean())
print('Average Freedom Score: ',data.Freedom.mean())
filtr=data['Happiness_Score']>data.Happiness_Score.mean()
data[filtr]
filtr2 = data['Freedom']>data.Freedom.mean()
data[filtr2].head(20)
filtr1=data['Happiness_Score']>data.Happiness_Score.mean()
filtr2=data['Freedom']<data.Freedom.mean()
data[filtr1 & filtr2]
filtr3=data['Freedom']>data.Freedom.mean() # data.Freedom.mean() is average freedom score
filtr4=data['Happiness_Score']<data.Happiness_Score.mean()
data[filtr3 & filtr4].head(10)
#tail() - for last lines. Default number is 5 but we want to write 10.
# [::-1] from end to top
data.tail(10)[::-1]        
       
# Average Happiness Score --- Long way
average_h = sum(data.Happiness_Score)/len(data.Happiness_Score)
# Average Happiness Score --- Short Way
# *** average_h = data.Happiness_Score.mean() ***
data['Who_is_happy'] = ['Happy' if i>average_h else 'Unhappy' for i in data.Happiness_Score]
data.loc[70:80,['Country','Who_is_happy']]



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
data = pd.read_csv('../input/world-happiness/2015.csv')
data.info()
data.describe()
data.columns
data.corr()
data.head(10)
data[data['Country']=='Turkey']
#fix the columns name
data.columns=[each.split()[0]+"_"+ each.split()[1] if (len(each.split())>1) else each for each in data.columns]
#rename columns name
data=data.rename(columns={'Economy_(GDP':'Economy_GDP'})
data=data.rename(columns={'Health_(Life':'Health_Life'})
data=data.rename(columns={'Trust_(Government':'Trust_Government'})
#max generosity value
#data.loc[data['Generosity'].idxmax()]
data[data['Generosity']==data['Generosity'].max()]
#min value of freedom
data[data['Freedom']==data['Freedom'].min()]
data.columns
#correlation plot
f,ax = plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(),annot=True , linewidths=.1 , fmt='.1f', ax=ax)
plt.show()
data.Freedom.plot(kind='line',label='freedom',grid=True,linewidth=1,linestyle='-')
data.Family.plot(label='family',grid=True,linewidth=1,linestyle=':',alpha=0.7)
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.show()
plt.scatter(data.Happiness_Score,data.Economy_GDP,color='red',alpha=0.5)
plt.xlabel('Happiness Score')
plt.ylabel('Economy')
plt.show()
data.Happiness_Score.plot(kind='hist',bins=30,figsize=(10,10))
plt.show()
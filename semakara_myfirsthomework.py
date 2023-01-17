# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/outbreaks.csv')
data.columns
data.info()

data.corr()
data.describe()
f,ax=plt.subplots(figsize=(13,13))
sns.heatmap(data.corr(),annot=True,fmt='0.1f',linewidths=4,ax=ax)
plt.show()
data.head(8)
data.Illnesses.plot(kind='line',label='Illnesses',color='g',linewidth=2,alpha=0.6,grid=True,linestyle='-')
plt.legend('lower right')
plt.xlabel('xaxis')
plt.ylabel('illnesses')
plt.title('Illnessesline')
plt.show()
data.plot(kind='scatter',x='Hospitalizations',y='Illnesses',color='b',alpha=0.5)
plt.xlabel('Hospitalizations')
plt.ylabel('Illnesses')
plt.title('Hosp vs Ill')
plt.scatter(data.Illnesses,data.Fatalities,color='r',alpha=0.7,)
plt.xlabel('Illnesses')
plt.ylabel('Fatalities')
plt.title('ill vs fatal')
plt.show()
data.Fatalities.plot(kind='hist',bins=100,figsize=(10,10),grid=True)
plt.show()
series=data['State']
print(series)
dataframe=data[['Location']]
print(dataframe)
data[np.logical_and(data['Location']=='Restaurant',data['Food']=='Fish' )]
data.head()
data[(data['Location']=='Restaurant') & ((data['Food']=='Lasagna')|(data['Food']=='Eggs'))]
data[(data['Species']=='Salmonella enterica')&((data['Illnesses']>0) & (data['Year']>2010))]
for i,v in enumerate(data['Status']):
    print(i,":",v)
for index,value in data[['Status']][3:4].iterrows():
    print(index,"=",value)

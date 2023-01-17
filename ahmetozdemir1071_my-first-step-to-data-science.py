# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/usa-cers-dataset/USA_cars_datasets.csv")
data.head()
data.tail()
data.info()

data.describe()
data_corr=data.corr()
f,ax=plt.subplots(figsize=(10,10))
sns.heatmap(data_corr,annot=True)
plt.show()
x=data.year
y=data.price
plt.scatter(x,y,color='purple',alpha=0.7)
plt.xlabel("YEAR")
plt.ylabel("PRICE")
plt.title("YEAR - PRICE")
plt.show()
colors=np.array(data.color.unique())
colorArr=np.array([colors,np.zeros(len(colors))])
colorArr[0,]

#find the frequencies of all colors
for i in data.color: 
    a=0
    for j in colorArr[0,]:
        if j==i:
            colorArr[1,a]+=1
        a+=1
other=0
colorDict={}
b=0
#I called the frequency less than twenty as "other"
for x in colorArr[1,]:
    if x>=20:
        colorDict.update({(colorArr[0,b]):x})
    else:
        other+=x      
    b+=1
colorDict.update({"other":other})        
plt.figure(figsize=(10,10))
plt.bar(colorDict.keys(),colorDict.values(),color="r",edgecolor="blue",linewidth=3)
plt.xlabel("COLOR")
plt.ylabel("FREQUENCY")
plt.title("Frequency of car color")
plt.show()

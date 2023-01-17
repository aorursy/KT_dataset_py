# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

import datetime as dt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/EdStatsData.csv")
data.info()
data.head(10)
data.tail(10)
data.columns
data.corr()
#correlation map
f,ax = plt.subplots(figsize=(25, 25))
sns.heatmap(data.corr(), annot=False, linewidths=.3, fmt= '.1f',ax=ax)
plt.show()
data.describe()
data["Country Name"].unique()
dataTurkey = data[data["Country Name"] == "Turkey"]
dataTurkey.head(15)

yearDic = {}

startYear =0
now = dt.datetime.now()

cols = dataTurkey.columns
for columnName in cols:
    try:
        year = int(columnName)
        if startYear == 0:
            startYear = year
        if year - startYear == 10 or now.year-1 == year:
            yearDic[str(startYear)]=str(year)
            startYear = year
    except:
        print("")    

for key,value in yearDic.items():
    print(key," : ",value)
  

i=0
for key,value in yearDic.items():
    dataTurkey.plot(kind='line', x=key, y=value,alpha = 1,color = 'C'+str(i),label=key+" - "+value,grid=True)
    i=i+1
plt.show()
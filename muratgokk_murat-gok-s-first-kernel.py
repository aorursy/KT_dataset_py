# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
data=pd.read_csv("/kaggle/input/videogamesales/vgsales.csv")


data.info()
data.columns

data.corr()

data.NA_Sales.plot(kind="line",color="g",label="NA_Sales",linewidth=9,alpha=0.3,grid=True,linestyle='solid')
data.EU_Sales.plot(kind="line",color="b",label="EU_Sales",linewidth=9,alpha=0.3,grid=True,linestyle="dashed")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("Line Plot ")

data.plot(kind="scatter",x="NA_Sales",y="EU_Sales",alpha=0.05,color="red")
plt.xlabel("NA Sales")
plt.ylabel("EU Sales")
plt.title("Scatter plot between NA-EU Sales")
data.Year.plot(kind="hist",bins=75,figsize=(18,9))
data["NA_Sales"].head()
data["Platform"].head(20)
data["Genre"].head(20)
import seaborn as sns
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.9, fmt= '.1f',ax=ax)
plt.show()

maxi =data["EU_Sales"].max()
print(maxi)
maxina=data["NA_Sales"].max()
print(maxina)
myear=data["Year"].max()
print(myear)
data[(data["EU_Sales"]>0.5)& (data["Year"]>2015)]



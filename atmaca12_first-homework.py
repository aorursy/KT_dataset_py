# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/videogamesales/vgsales.csv")
data.head()
data.tail()
data.columns
data.info()
data.dropna(inplace=True)
data.Year=data.Year.astype(int)
data.head()
#sales figures in america by years histogram

data.groupby("NA_Sales")["Year"].mean().plot(kind="hist",color="r", alpha=0.5,grid=True,figsize=(8,5))

plt.xlabel("YEARS")

plt.ylabel("GAMES(millions)")

plt.title("sales figures in america by years")



#NA_Sales EU_Sales Scatter Plot

data.plot(kind="scatter",x="NA_Sales",y="EU_Sales",grid=True,marker="x",figsize=(8,5))

plt.xlabel('NA_Sales')              # label = name of label

plt.ylabel('EU_Sales')

plt.title('NA_Sales EU_Sales Scatter Plot') 
#NA_Sales JP_Sales Scatter Plot

data.plot(kind="scatter",color='g',x="NA_Sales",y="JP_Sales",grid=True,marker="o",figsize=(8,5))

plt.xlabel('NA_Sales')              # label = name of label

plt.ylabel('JP_Sales')

plt.title('NA_Sales JP_Sales Scatter Plot') 
#NA_Sales Global_Sales Scatter Plot

data.plot(kind="scatter",color='b',x="NA_Sales",y="Global_Sales",grid=True,marker="<",figsize=(8,5))

plt.xlabel('NA_Sales')              # label = name of label

plt.ylabel('Global_Sales')

plt.title('NA_Sales Global_Sales Scatter Plot') 
#popular game types in america by years

data.groupby(["Genre","Year"])["NA_Sales"].mean().plot(kind="line",color="r", alpha=0.5,grid=True,figsize=(15,10))

plt.xlabel("YEARS")

plt.ylabel("GAMES")

plt.title("")

#sales figures in europe by years histogram

data.groupby("EU_Sales")["Year"].mean().plot(kind="hist",alpha=1,grid=True,figsize=(8,5))

plt.xlabel("YEARS")

plt.ylabel("GAMES(millions)")

plt.title("sales figures in europe by years")
#EU_Sales NA_Sales Scatter Plot

data.plot(kind="scatter",x="EU_Sales",y="NA_Sales",grid=True,marker="x",figsize=(8,5))

plt.xlabel('EU_Sales')              # label = name of label

plt.ylabel('NA_Sales')

plt.title('EU_Sales NA_Sales Scatter Plot') 
#EU_Sales JP_Sales Scatter Plot

data.plot(kind="scatter",x="EU_Sales",y="JP_Sales",grid=True,marker="o",figsize=(8,5))

plt.xlabel('EU_Sales')              # label = name of label

plt.ylabel('JP_Sales')

plt.title('EU_Sales JP_Sales Scatter Plot') 
#EU_Sales Global_Sales Scatter Plot

data.plot(kind="scatter",color='b',x="EU_Sales",y="Global_Sales",grid=True,marker="<",figsize=(8,5))

plt.xlabel('EU_Sales')              # label = name of label

plt.ylabel('Global_Sales')

plt.title('EU_Sales Global_Sales Scatter Plot') 
data.groupby(["Genre","Year"])["EU_Sales"].mean().plot(kind="line",color="r", alpha=0.5,grid=True,figsize=(15,10))

plt.xlabel("YEARS")

plt.ylabel("GAMES")

plt.title("")

#sales figures in japan by years histogram

data.groupby("JP_Sales")["Year"].mean().plot(kind="hist",alpha=1,grid=True,figsize=(8,5))

plt.xlabel("YEARS")

plt.ylabel("GAMES(millions)")

plt.title("sales figures in japan by years")
#JP_Sales NA_Sales Scatter Plot

data.plot(kind="scatter",x="JP_Sales",y="NA_Sales",grid=True,marker="x",figsize=(8,5))

plt.xlabel('JP_Sales')              # label = name of label

plt.ylabel('EU_Sales')

plt.title('JP_Sales NA_Sales Scatter Plot') 
#JP_Sales EU_Sales Scatter Plot

data.plot(kind="scatter",x="JP_Sales",y="EU_Sales",grid=True,marker="o",figsize=(8,5))

plt.xlabel('JP_Sales')              

plt.ylabel('EU_Sales')

plt.title('JP_Sales EU_Sales Scatter Plot') 
#JP_Sales Global_Sales Scatter Plot

data.plot(kind="scatter",color='b',x="JP_Sales",y="Global_Sales",grid=True,marker="<",figsize=(8,5))

plt.xlabel('JP_Sales')              

plt.ylabel('Global_Sales')

plt.title('JP_Sales Global_Sales Scatter Plot') 
data.groupby(["Genre","Year"])["JP_Sales"].mean().plot(kind="line",color="r", alpha=0.5,grid=True,figsize=(15,10))

plt.xlabel("YEARS")

plt.ylabel("GAMES")

plt.title("")
data
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.plot(kind="scatter", x="Year" , y="Global_Sales",alpha=1,grid=True ,marker="o",figsize=(8,5))
#Histogram showing global game sales by years 

data.groupby("Global_Sales")["Year"].mean().plot(kind="hist",color='g',figsize=(8,5),grid=True)

plt.xlabel("YEARS")

plt.ylabel("GAMES")

plt.title("showing global game sales by years ")
#games released by years

data.plot(kind="scatter", x="Genre",y="Year",alpha=1,marker="o",grid=True,figsize=(15,8))

plt.xlabel("GENRE")

plt.ylabel("YEAR")
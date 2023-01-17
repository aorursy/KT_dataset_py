# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("../input/turkey-corona-data/turkey_corona_data.csv")

df.head()
df["Date"]=pd.to_datetime(df["Date"])

df["Year"]=df["Date"].dt.year

df["Month"]=df["Date"].dt.month

df["Day"]=df["Date"].dt.day
df.head()
month=["January","February","March","April","May","June","July","August","September","October","November","December"]

df["Month"].replace(np.arange(1,13),np.array(month),inplace=True)



df.index=df["Date"]

df.index=pd.DatetimeIndex(df.index)

df.drop(["Date"],axis=1,inplace=True)
df.head()
df.tail()
df.describe().T
sns.set(style="whitegrid")
fig,ax=plt.subplots(nrows=3,ncols=2,figsize=(16,15))

list_select=["Confirmed","Deaths","Recovered"]

color=["#FF0000","#8E44AD","#008000"]

count=0

for j in range(3):

    sns.distplot(df[list_select[count]],ax=ax[j][0],kde=False,color=color[count],bins=15)

    sns.kdeplot(df[list_select[count]],ax=ax[j][1],shade=True,color=color[count])

    count+=1
fig,ax=plt.subplots(nrows=2,ncols=1,figsize=(16,10))

sns.countplot(df["Day"],ax=ax[0])

sns.countplot(df["Month"],ax=ax[1]);
list_select=['Confirmed', 'Deaths', 'Recovered']
plt.figure(figsize=(16,7))

plt.plot(df["Confirmed"],lw=5,color="#FF0000")

plt.plot(df["Deaths"],"--",lw=5,color="#8E44AD")

plt.plot(df["Recovered"],"-.",lw=5,color="#008000")

plt.legend(list_select);
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(16,5))

sns.lineplot(x=df["Month"],y=df["Deaths"],ax=ax[0],color="#8E44AD")

ax[0].set_title("Deaths-Month",fontsize=15)

sns.lineplot(x=df["Month"],y=df["Recovered"],ax=ax[1],color="#008000")

ax[1].set_title("Recovered-Month",fontsize=15);
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(16,5))

sns.pointplot(x=df["Month"],y=df["Deaths"],ax=ax[0],color="#8E44AD")

ax[0].set_title("Deaths-Month",fontsize=15)

sns.pointplot(x=df["Month"],y=df["Recovered"],ax=ax[1],color="#008000")

ax[1].set_title("Recovered-Month",fontsize=15);
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(16,5))

sns.stripplot(x=df["Month"],y=df["Deaths"],ax=ax[0])

ax[0].set_title("Deaths-Month",fontsize=15)

sns.stripplot(x=df["Month"],y=df["Recovered"],ax=ax[1])

ax[1].set_title("Recovered-Month",fontsize=15);
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(16,5))

sns.swarmplot(x=df["Month"],y=df["Deaths"],ax=ax[0])

ax[0].set_title("Deaths-Month",fontsize=15)

sns.swarmplot(x=df["Month"],y=df["Recovered"],ax=ax[1])

ax[1].set_title("Recovered-Month",fontsize=15);
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(16,5))

sns.boxplot(y=df["Deaths"],ax=ax[0],color="#8E44AD")

ax[0].set_title("Deaths-Month",fontsize=15)

sns.boxplot(y=df["Recovered"],ax=ax[1],color="#008000")

ax[1].set_title("Recovered-Month",fontsize=15);
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(16,5))

sns.boxplot(x=df["Month"],y=df["Deaths"],ax=ax[0])

ax[0].set_title("Deaths-Month",fontsize=15)

sns.boxplot(x=df["Month"],y=df["Recovered"],ax=ax[1])

ax[1].set_title("Recovered-Month",fontsize=15);
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(16,5))

sns.violinplot(x=df["Month"],y=df["Deaths"],ax=ax[0])

ax[0].set_title("Deaths-Month",fontsize=15)

sns.violinplot(x=df["Month"],y=df["Recovered"],ax=ax[1])

ax[1].set_title("Recovered-Month",fontsize=15);
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(16,5))

sns.boxenplot(y=df["Deaths"],ax=ax[0],color="#8E44AD")

ax[0].set_title("Deaths-Month",fontsize=15)

sns.boxenplot(y=df["Recovered"],ax=ax[1],color="#008000")

ax[1].set_title("Recovered-Month",fontsize=15);
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(16,5))

sns.boxenplot(x=df["Month"],y=df["Deaths"],ax=ax[0])

ax[0].set_title("Deaths-Month",fontsize=15)

sns.boxenplot(x=df["Month"],y=df["Recovered"],ax=ax[1])

ax[1].set_title("Recovered-Month",fontsize=15);
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(16,5))

sns.barplot(y=df["Deaths"],ax=ax[0],color="#8E44AD")

ax[0].set_title("Deaths-Month",fontsize=15)

sns.barplot(y=df["Recovered"],ax=ax[1],color="#008000")

ax[1].set_title("Recovered-Month",fontsize=15);
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(16,5))

sns.barplot(x=df["Month"],y=df["Deaths"],ax=ax[0])

ax[0].set_title("Deaths-Month",fontsize=15)

sns.barplot(x=df["Month"],y=df["Recovered"],ax=ax[1])

ax[1].set_title("Recovered-Month",fontsize=15);
sns.pairplot(df,vars=list_select,aspect=1.6);
sns.pairplot(df,vars=list_select,aspect=1.6,kind="reg");
sns.pairplot(df,vars=list_select,aspect=1.6,kind="reg",hue="Month");
fig,ax=plt.subplots(ncols=2,nrows=1,figsize=(16,8))

sns.scatterplot(x=df["Confirmed"],y=df["Deaths"],ax=ax[0],color="#8E44AD")

sns.scatterplot(x=df["Confirmed"],y=df["Recovered"],ax=ax[1],color="#008000");
fig,ax=plt.subplots(ncols=2,nrows=1,figsize=(16,8))

sns.scatterplot(x=df["Confirmed"],y=df["Deaths"],hue=df["Month"],ax=ax[0])

sns.scatterplot(x=df["Confirmed"],y=df["Recovered"],hue=df["Month"],ax=ax[1]);
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
#Importing some important visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt

games=pd.read_csv("/kaggle/input/videogamesales/vgsales.csv")#It is a csv file so I used .read_csv()
games.info() #It shows number and types of each features
games.describe()#We can see each columns's statistical values by .describe() method
games.head()
games.tail()
games.corr() # It shows correlations between features
#We can see that there is a strong correlation between global sales and sales in Europe and Norht America
# Also there is a positive correlation between sales in Europe and sale in North America
#Let's show this correlation by using heatmap
f,ax=plt.subplots(figsize=(18,18))
sns.heatmap(games.corr(),annot=True,cmap="coolwarm",linewidth=0.5,fmt=".1f",ax=ax,vmin=-1.0,vmax=1.0)
#Line plot
games_global=games.groupby("Year")["Global_Sales"].sum()
games_NA=games.groupby("Year")["NA_Sales"].sum()
games_global.plot(kind="line",x="Year",y="Global_Sales",grid=True,color="red",alpha=0.5,linewidth=1,figsize=(10,10))
games_NA.plot(kind="line",x="Year",y="Global_Sales",grid=True,color="blue",alpha=0.5,linewidth=1,figsize=(10,10))
plt.legend(loc="upper left")
plt.xlabel("YEAR")
plt.ylabel("GLOBAL SALES")
plt.show()
#When we looked this data, we can see correlation between sales of North America and globals sales. They have same trend.
#Scatter plot
games_sales=games.groupby("Year")["EU_Sales","NA_Sales"].sum()
games_sales.plot(kind="scatter",x="EU_Sales",y="NA_Sales",color="Blue",alpha=0.5)
#We can see strong positive correlation between sales of Europe and North America

#Histogram

#This function will center labels in histogram

games_genre=games["Genre"].value_counts()
df_genre=pd.DataFrame(games_genre).reset_index().rename(columns={"index":"Genre","Genre":"Total_Number"})
df_genre.plot.bar(x="Genre",y="Total_Number")
# We can see that number of action games is much higher than other genres
 
platforms=games["Platform"].value_counts()
df_platforms=pd.DataFrame(platforms).reset_index().rename(columns={"index":"Platform","Platform":"Total_Number"})
df_platforms.plot.bar(x="Platform",y="Total_Number",figsize=(15,15),color="red")
# We can see that most of the games belongs to PS2 and DS platforms
platformsYears=games.groupby("Year")["Platform"].value_counts()
df_PY=pd.DataFrame(platformsYears).rename(columns={"Year":"Year","Platform":"Platform","Platform":"Count"}).reset_index()
df_PY.head()
platforms=df_PY["Platform"].unique()
f,ax=plt.subplots(figsize=(18,18))
for i in platforms:
    special_data=df_PY[df_PY["Platform"]==i]
    special_data.plot(kind="line",x="Year",y="Count",label=i,grid=True,alpha=0.5,linewidth=1,ax=ax)
    plt.legend(loc="upper right")
    plt.xlabel("Year")
    plt.ylabel("Number of Games")
plt.show()
#This graph shows number of games each platfrom released in each year.
games.head()
#I will try to find total global sales of platforms each year 

#First I will form a dataframe that contains year, platform and total sales of that platform
years=games["Year"].unique()
years= np.sort(years[np.logical_not(np.isnan(years))]) #Dropping nan years form array
platforms=games["Platform"].unique()
platforms_sales=pd.DataFrame(columns=["Year","Platform","Sales"])
for y in years:
    for p in platforms:
        sales=games[(games["Year"]==y)&(games["Platform"]==p)]["Global_Sales"].sum()
        new_frame=pd.DataFrame([[y,p,sales]],columns=["Year","Platform","Sales"])
        platforms_sales=platforms_sales.append(new_frame,ignore_index=True)
f,ax=plt.subplots(figsize=(18,18))
for i in platforms:
    special_data=platforms_sales[platforms_sales["Platform"]==i]
    special_data.plot(kind="line",x="Year",y="Sales",label=i,grid=True,alpha=0.5,linewidth=1,ax=ax)
    plt.legend(loc="upper right")
    plt.xlabel("Year")
    plt.ylabel("Total Sales")
    plt.title("Total Global Sales with respect to Platforms")
plt.show()
games.info()
# We can see that numbers of publishers and years are less than 16598 so we have some NaN values
games[games["Year"].isnull()==True]#We found NaN year values

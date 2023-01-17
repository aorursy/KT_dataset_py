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
df=pd.read_csv('/kaggle/input/tv-shows-on-netflix-prime-video-hulu-and-disney/tv_shows.csv')
#Exploratory data analysis
print(df.shape)
print(df.head())
print(df.columns)

print(df.info())
#Age contains a lot of missing values , replacing those values by 'all'
df.Age.unique()
df.Age.value_counts(dropna=False)
df.Age=df.Age.replace(np.nan,"all")
df.Age.value_counts(dropna=False)
#Again now for the imdb ratings in the dataset
df.IMDb.unique()
df.IMDb.value_counts(dropna=False)
df.IMDb=df.IMDb.replace(np.nan,"0")
df.IMDb.isna().sum()
#This implies there are no null values present in the column of the dataset.
df["Rotten Tomatoes"].unique()
#Since the rotten tomatoes ratings contain a lot of missing values so remove this column from the dataset
df=df.drop("Rotten Tomatoes",axis=1)
df.columns
df=df.drop("Unnamed: 0",axis=1)
#As type also contains all the rows containing ones , so removing type from the data
df=df.drop("type",axis=1)
#Now checking the data types of the columns , we get
df.dtypes
#Here IMDb ratings are object type but they should be numeric
df.IMDb=pd.to_numeric(df.IMDb)
df2=pd.melt(df,id_vars=[ 'Title', 'Year', 'Age', 'IMDb'],var_name="All platforms")
df2=df2[df2["value"]==1]
df2=df2.drop("value",axis=1)
df2["All platforms"].value_counts()
fig = plt.figure(figsize=(12, 12))
sns.countplot(x='All platforms',data=df2)
plt.title("Number of shows on each platform")
plt.show()
#This tells us the show is available on which platforms
str=input("Enter the name of the show: ").lower()
index=[]
for i in range(df2.shape[0]):
    if df2.iloc[i,0].lower()==str:
        index.append(i)
if len(index)==0:
    print("The show is not available on Netfix, Prime Video ,Hulu and Disney+.")
else:
    for i in range(len(index)):    
        print("The show is available on {}".format(df2.iloc[index[i],4]))
#IMDb rating ,year of the show
str=input("Enter the name of the show: ").lower()
index=[]
for i in range(df2.shape[0]):
    if df2.iloc[i,0].lower()==str:
        index.append(i)
if len(index)==0:
    print("The show is not available on Netfix, Prime Video ,Hulu and Disney+.")
else:  
        print("The IMDb rating of {} is {}. The show was released in the year {}.".format(str,df2.iloc[index[0],3],df2.iloc[index[0],1]))
#Analysis of age groups vs streaming platform
sns.set_style("darkgrid")
fig = plt.figure(figsize=(12, 12))
sns.countplot(x="All platforms",hue="Age",data=df2)
plt.title("Analysis of age groups vs streaming platform")
plt.ylabel("Number of shows")
plt.xlabel("Platforms")
plt.show()
#Comparing the ratings of the shows
fig = plt.figure(figsize=(12, 12))
y=sns.countplot(x="IMDb",data=df2)
plt.xticks(rotation=-45,fontsize=6)
plt.xlabel("Ratings")
plt.ylabel("Number of shows")
plt.title("Comparing the ratings of the shows")
plt.show()
#Number of platforms a show is present on
frequency_of_shows=df2.Title.value_counts()
frequency_of_shows=pd.DataFrame(frequency_of_shows)
fig = plt.figure(figsize=(12, 12))
sns.countplot(x="Title",data=frequency_of_shows)
plt.title("Number of platforms a show is present on")
plt.xlabel("Number of platforms")
plt.ylabel("Number of shows")
plt.show()
#Number of shows in a various years
for i in range(frequency_of_shows.shape[0]):
    frequency_of_shows.loc[frequency_of_shows.index[i],"Years"]=df2.loc[df2[df2.Title==frequency_of_shows.index[i]].index[0],"Year"]
frequency_of_shows.Years=frequency_of_shows.Years.astype(np.int64)
fig = plt.figure(figsize=(12, 12))
sns.countplot(x="Years",data=frequency_of_shows)
plt.xticks(rotation=-45,fontsize=6)
plt.title("Number of shows in a previous years")
plt.xlabel("Year")
plt.ylabel("Number of shows")
plt.show()
#Thank You
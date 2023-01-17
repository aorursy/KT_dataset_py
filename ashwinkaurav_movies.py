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
data=pd.read_csv("/kaggle/input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv")
data.head()
print(data.info())

data.drop(["Unnamed: 0","ID"],axis=1,inplace=True)

#Getting columns with missing data

cols_with_nan=[col for col in data.columns if data[col].isnull().any()]

import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline
fig, ax = plt.subplots(figsize=(15,6))

sns.barplot(data.dropna(axis=0,subset=["IMDb"],inplace=False)["Year"],data.dropna(axis=0,subset=["IMDb"],inplace=False)["IMDb"],ax=ax)
new_data=data.dropna(subset=["IMDb"])

new_data.corr()






list_platforms=["Netflix","Hulu","Prime Video","Disney+"]

fig,(ax1,ax2,ax3,ax4)=plt.subplots(1,4,sharey=True,figsize=(12,6))

fig.subplots_adjust(hspace=0, wspace=0)

ax1.boxplot(new_data["IMDb"][new_data["Netflix"]==1])

ax1.title.set_text("Netflix")

ax2.boxplot(new_data["IMDb"][new_data["Hulu"]==1])

ax2.title.set_text("Hulu")

ax3.boxplot(new_data["IMDb"][new_data["Prime Video"]==1])

ax3.title.set_text("Prime Video")

ax4.boxplot(new_data["IMDb"][new_data["Disney+"]==1])

ax4.title.set_text("Disney+")

#REMOVING ROWS WITH NAN VALUES FOR GENRES AND DIRECTORS COLUMNS 



new_data.dropna(subset=["Genres","Directors"],axis=0,inplace=True)

#Getting unique values of genres,directors and countries.

genres=new_data["Genres"].unique()

directors=new_data["Directors"].unique()

country=new_data['Country'].unique()

print(genres,directors,country)

#But as we can observe data set each of these have more than a single entity for almost each row . To make sure we don't repeat them in our unique value list we will 

#pick only different values out.

print(len(genres),len(directors),len(country))

final_genre=[]

for i in range(len(genres)):

    lis=list(map(str,genres[i].split(",")))

    for j in lis:

        if j not in final_genre:

            final_genre.append(j)
final_genre
final_directors=[]

for i in range(len(directors)):

    lis=list(map(str,directors[i].split(",")))

    for j in lis:

        if j not in final_genre:

            final_directors.append(j)
len(final_directors)
#Merging the original data (preprocessed) with the new One Hot Encoded Genre DataFrame

final_data=new_data

to_add_into_df=[0]*new_data.shape[0]

for i in range(len(final_genre)):

    df=pd.DataFrame(to_add_into_df,columns=[final_genre[i]]) 

    final_data=pd.concat([final_data,df],axis=1)

    
final_data.fillna(0,inplace=True)

final_data["Genres"].replace(to_replace=0,value="None",inplace=True)
#Setting the values of the concated genre DataFrame columns to 1 if the given genre was originally in "Genres" column of the original Dataset

for i in range(len(final_genre)):#15

    for j in range(final_data.shape[0]):

        if final_genre[i] in final_data.iloc[j,11]:

            final_data.iloc[j,15+i]=1

    
plt.figure(figsize=(12,15))

for i in range(1,27):

    plt.subplot(15,2,i)

    plt.title(final_genre[i-1])

    plt.plot(final_data["IMDb"][final_data.iloc[:,15+i]==1])

mean_IMDb={}

standard_dev={}

total_count_of_genre={} #This is the count of number of movies of a specific genre

for i in range(len(final_genre)):

    mean_IMDb[final_genre[i]]=final_data["IMDb"][final_data.iloc[:,15+i]==1].mean()

    standard_dev[final_genre[i]]=final_data["IMDb"][final_data.iloc[:,15+i]==1].std()

    total_count_of_genre[final_genre[i]]=sum(final_data.iloc[:,15+i])
for i in range(len(final_genre)):

    print("Genre: ||",final_genre[i],"----MEAN: %.2f "%mean_IMDb[final_genre[i]],"----Standard Deviation: || %.2f"%standard_dev[final_genre[i]],"XX Total Count == %.0f  "%total_count_of_genre[final_genre[i]])
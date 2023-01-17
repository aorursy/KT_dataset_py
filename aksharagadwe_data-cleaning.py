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
data = pd.read_csv("/kaggle/input/netflix-shows/netflix_titles.csv")
data.info()
data = data.drop("date_added" , axis=1)
data.info()
data.type.value_counts()
data.country.value_counts()
genre = []
def genre_dict(string):
    x = string.split(", ")
    for genre_type in x:
        if genre_type not in genre:
            genre.append(genre_type)
for values in data.listed_in:
    genre_dict(values)
len(genre)
d ={}
for i in genre:
    d[i] = 0
d
data.assign(**d)
for i in range(data.shape[0]):
    temp = data.loc[i,"listed_in"]
    temp = temp.split(", ")
    for value in temp:
        data.at[i,value]=1
for types in genre:
    data[types] = data[types].fillna(0)
    data[types] = pd.to_numeric(data[types])
data.info()
data["Sci-Fi & Fantasy"] = 0
data["Horror"] = 0
data["Action & Adventure"] = 0
data["Mysteries"] = 0
data["Romantic"] = 0
data["Classic"] = 0
data["Cult"] = 0
for i in range(data.shape[0]):
    if data.loc[i,"TV Comedies"] == 1:
        data.at[i,"Comedies"] = 1
        
    if data.loc[i,"TV Dramas"] == 1:
        data.at[i,"Dramas"]=1
        
    if data.loc[i,"TV Thrillers"] == 1:
        data.at[i,"Thrillers"]=1
        
    if data.loc[i,"TV Mysteries"] == 1:
        data.at[i,"Mysteries"]=1
        
    if data.loc[i,"TV Sci-Fi & Fantasy"] == 1:
        data.at[i,"Sci-Fi & Fantasy"]=1
        
    if data.loc[i,"Stand-Up Comedy & Talk Shows"] == 1:
        data.at[i,"Stand-Up Comedy"]=1
        
    if data.loc[i,"Romantic TV Shows"] == 1:
        data.at[i,"Romantic"]=1
        
    if data.loc[i,"Romantic Movies"] == 1:
        data.at[i,"Romantic"]=1
        
    if data.loc[i,"Horror Movies"] == 1:
        data.at[i,"Horror"]=1
        
    if data.loc[i,"TV Horror"] == 1:
        data.at[i,"Horror"]=1
        
    if data.loc[i,"TV Action & Adventure"] == 1:
        data.at[i,"Action & Adventure"]=1
        
    if data.loc[i,"Classic & Cult TV"] == 1:
        data.at[i,"Classic"]=1
        data.at[i,"Cult"]=1
        
        
    if data.loc[i,"Classic Movies"] == 1:
        data.at[i,"Classic"]=1
    
        
    
data = data.drop(["TV Comedies","TV Dramas","TV Thrillers",
                 "TV Mysteries","TV Sci-Fi & Fantasy",
                 "Stand-Up Comedy & Talk Shows",
                 "Romantic TV Shows","Romantic Movies",
                 "Horror Movies","TV Horror","Classic Movies",
                 "TV Action & Adventure","Classic & Cult TV"],axis=1)
data.info()
data=data.drop("listed_in",axis=1)
data.type.value_counts()
data.rating.value_counts()
data = data.drop("duration" ,axis=1)
data.director.value_counts()
data.country.value_counts()
data["country"] = data["country"].fillna(0)
country = []
def country_dict(string):
    x = string.split(", ")
    for country_type in x:
        if country_type not in country:
            country.append(country_type)
for value in data.country:
    if value != 0:
        country_dict(value)
country
c= {}
for i in country:
    c[i]=0
data.assign(**c)
for i in range(data.shape[0]):
    temp = data.loc[i,"country"]
    if temp != 0:
        temp = temp.split(", ")
        for value in temp:
            data.at[i,value]=1
for types in country:
    data[types] = data[types].fillna(0)
    data[types] = pd.to_numeric(data[types])
data.info()
data = data.drop("country" , axis =1)
data.head()
data["director"] =  data["director"].fillna(0)

data = data.drop("description",axis =1 )
data.info()
data.rating.fillna("None")
data.rating.value_counts()
data["rating"] = data["rating"].astype('category')

data["ratings"]=data["rating"].cat.codes
data.ratings
data.ratings.value_counts()
data["ratings"] = data["ratings"] + 1
data["ratings"].value_counts()

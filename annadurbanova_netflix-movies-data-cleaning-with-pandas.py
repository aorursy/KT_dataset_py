# Author: Anna Durbanova

# Date: 28.09.2020



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from datetime import datetime

import os



%matplotlib inline
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

data=pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')
#data=pd.read_csv("netflix_titles.csv")

data["date_added"]=data["date_added"].str.replace(",", "").str.strip() ## Strip removes spaces front and back

data["date_added"]=pd.to_datetime(data["date_added"], format="%B %d %Y")



data["director"]=data["director"].fillna("Unknown")

data["cast"]=data["cast"].fillna("Unknown")

data["country"]=data["country"].fillna("Unknown")





data["Season"]=""



columns_name=["show_id", "type", "title", 

              "director", "cast", "country",

              "date_added", "release_year", "rating",

              "duration", "Season", "listed_in", "description"]

data=data.reindex(columns=columns_name)



data["Season"] = data[data["duration"].str.contains("Season")]["duration"] ## Separating Season from duration



## Make a Season Column an Integer



data["Season"]=data["Season"].fillna("0")

data["Season"]=data["Season"].str.replace("Season", "").str.replace("s", "")

data["Season"]=data["Season"].astype(str).astype(int)



## Make a duration column an Integer

data["duration"]=data.duration.str.replace('^(\d+)(.Seasons*)$', "0") ## Remove The Season from the column

data["duration"]=data["duration"].str.replace(" min", "") ## Remove min

data["duration"]=data["duration"].astype(int) ## Convert to Integer







data
def missing_values(n):

    df=pd.DataFrame()

    df["missing_values, %"]=data.isnull().sum()*100/len(data.isnull())

    df["missing_values, sum"]=data.isnull().sum()

    return df.sort_values(by="missing_values, %", ascending=False)

missing_values(data)
## After all cleaning, there is not so much of left NAs. 
data["type"].value_counts()
(data["listed_in"].str.contains("Horror")).sum() # 316 horror Movies & TV -shows

((data["listed_in"].str.contains("Horror"))[(data["type"]=="Movie")]).sum()
((data["listed_in"].str.contains("Horror"))[(data["type"]=="TV Show")]).sum()
data[data["listed_in"].str.contains("Horror")][data["type"]=="TV Show"].head(5) ## The list of 5 TV Shows
mask = data["listed_in"].str.contains("Horror")

(data[mask]

.groupby("title")

 [['type', 'title', 'country','description']]

.sum()

.head(10)

) 



## First 10 horror movies
sort= data["duration"]!=0

data[sort]["duration"].median()



## On average 98 minutes
data["duration"].max() # Min
data[data["duration"]==312]



## The longest Movie was Black Mirror Bandersnatch
data["Season"].max()
data[data["Season"]==15]



## Two movies: Grey's Anatomy and NCIS, one is romantic, another one is a crime TV
sort_s= data["Season"]!=0

data[sort_s]["Season"].median() ## -- On average 1 Season
mask=data['type']=="Movie"

data[mask].sort_values(by="release_year", ascending= True).head(5)

## And two from 1942 : Prelude to War and The Battle of Midway -- all documentaries
## What is the oldest TV-Show on Netflix

mask2=data['type']=="TV Show"

data[mask2].sort_values(by="release_year", ascending=True).head(5)



## The oldest TV Show was from 1925, called "PioneerS: First Women Filmmakers"

(data[data["type"]=="Movie"]

.groupby("country")

[["show_id"]]

.count()

.sort_values(by="show_id", ascending=False)

.head(10)

)

## Most of the movies are made by America, Then India
## Countries and Tv-Show releases 

(data[data["type"]=="TV Show"]

.groupby("country")

[["show_id"]]

.count()

.sort_values(by="show_id", ascending=False)

.head(10)

)

## Most of the TV Shows are made by USA
data["rating"].value_counts()

## Most of the movies are # TV-MA	Suitable for Mature Audiences Only  and Good for 14 years old kids.



# TV-MA	Suitable for Mature Audiences Only 

# TV-14 Programs rated TV-14 contains material that parents or adult guardians may find unsuitable for children under the age of 14

# TV - PG This program is intended to be viewed by mature, adult audiences and may be unsuitable for children under 17.

# R - Restricted, Children Under 17 Require Accompanying Parent or Adult Guardian. 

# PG-13 -  Parents Strongly Cautioned. Some Material May Be Inappropriate For Children Under 13

# NR - Not Rated

# PG - Parental Guidance General viewing

# TV-Y7 - This program is most appropriate for children age 7 and up.

#TV - G - This program is suitable for all ages.

# TV-Y  - this program is aimed at a very young audience, including children from ages 2-6

# fantasy violence may be more

# TV-Y7-FV  - directed to older children with intense or more combative than other programs

# G - General audiences – All ages admitted.

# UR - Under Rated

# NC-17 - No One 17 and Under Admitted.
(data["cast"].str.contains("Tom Cruise")).sum()

data[data["cast"].str.contains("Tom Cruise")]



## One of my favorite actors are listed too, Rain Main and Magnolia with Tom Cruise :)
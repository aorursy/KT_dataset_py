# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import seaborn as sns



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#reading in the data 



ratings_data = "../input/ramen-ratings/ramen-ratings.csv"



ratings1 = pd.read_csv(ratings_data)



#changed the index to the review number because it's already a useful indexing feature

ratings = ratings1.set_index("Review #")

ratings.head()
#Let's group them by country and see what happens 



ratings.groupby("Country").tail()
#How many countries are represented? 



unique = ratings.Country.unique() #This provides a list of countries



len(unique) #just tells us the number of items in the list

ratings = ratings.rename(columns = {"Stars": "Ratings"}) #I keep calling the stars ratings so I'm going to make life easier for me

#I needed to reassign this function to the original table

print(ratings.columns) # to check if it's worked - it does!
ratings.sort_values(by=["Country"], inplace=True, ascending=True)

#ratings.sort_values(by=['Ratings'], inplace=True, ascending=False) #sorts values 

ratings



#Now we should have the lowest at the bottom. We also need to get rid of the unrated restaurants because they're 

#unhelpful. 

ratings = ratings.drop([2458, 2548, 1587]) #There's likely a more efficient way to do this. I'll figure it out another time.
#The values in the ratings column were just objects! Not numeric values that I couldn't do anything with.

# What jokers. 



country_group = ratings.groupby(["Country", "Ratings"])

#country_group.Ratings.to_numeric()

ratings.Ratings = pd.to_numeric(ratings.Ratings)

#I've tried creating a heatmap. For me, it doesn't really make sense. 



#Let's create a series that determines which country has the highest average rating.



country_group = ratings.groupby(["Country"])

country_average = country_group.Ratings.mean().sort_values(ascending=False)



#IT FINALLY WORKED. YOU HAVE NO IDEA HOW ANNOYING THIS WAS.
#Now let's try to create a useful  chart - swarm plot? 



plt.figure(figsize=(25,10))

plt.title("Countries versus rating")

sns.set_style("white")

sns.swarmplot(y=ratings["Country"], x=ratings["Ratings"])

plt.xlabel("Ratings")

plt.ylabel("Country")

plt.legend(loc="upper right")



#This is a useless viz.
#Bar chart time 



plt.figure(figsize=(45,20))

plt.title("Countries versus rating")

sns.set_style("white")

ratings.sort_values("Country", inplace=True)

sns.barplot(y=ratings.Ratings, x=ratings.Country)
ca = country_average.to_frame()

ca = ca.reset_index()

ca.head()
ca = ca.reset_index(drop=True) #I was getting many errors without dropping the index

plt.figure(figsize=(45,20))

plt.title("Countries versus rating")

sns.set_style("white")

sns.barplot(y=ca["Ratings"], x=ca["Country"])
#time to check if I've done something dumb to the original data set 

ratings.tail()



#It seems fine. It's just grouped by country and the ratings are now values
#What's next - let's see what brand has the best rating? 



ratings.Brand.value_counts()



#there are 355 unique brands, I'm not sure how to visualise that in any meaningful way
ratings.Style.value_counts()



# I can do something with the style of ramen though.
#get average ratings for the style of ramen 



style_group = ratings.groupby(["Style"])

style_average = style_group.Ratings.mean().sort_values(ascending=False)

sa = style_average.to_frame().reset_index()

sns.barplot(y=sa.Style, x=sa.Ratings, palette="Blues_d")

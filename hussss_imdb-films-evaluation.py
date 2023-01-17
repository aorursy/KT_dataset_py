# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#reading the data

df = pd.read_csv("../input/IMDB-Movie-Data.csv") 
#first look on the data

df.head(3) 
#renaming some columns

df.rename(columns = {"Year": "Release_year", "Runtime (Minutes)": "Runtime_minutes", "Revenue (Millions)": "Revenue_millions"}, inplace = True)
#column names

df.columns
#information about columns(dtypes, non-null values,...)

df.info()
#Correlations among columns 

df.corr()
#as we can see, highest correlations is 0.6

#the relationships that have highest correlation coefficients are; Rating-metascore and Votes-Revenue.

f,ax = plt.subplots(figsize = (12, 12))

sns.heatmap(df.corr(), annot = True, linewidths = .5, fmt = '.1f', ax = ax)
#top 5 films that have highest revenue

df2 = df.sort_values(by = "Revenue_millions", ascending = False).loc[:, ["Title", "Release_year", "Rating", "Revenue_millions"]].head(5)

df2
#pivot table

#we could get information about the average of ratings and runtimes, sum of revenues and number of films, by applying pivot table, on a release year basis.

df.pivot_table(df, index = ["Release_year"], aggfunc = {"Title": "count", "Rating": "mean", "Votes": "sum", "Revenue_millions": "sum", "Runtime_minutes": "mean"})
#creating color palette

palette = sns.cubehelix_palette(n_colors = 10, start = 2.9, rot =.4)
#top 5 films that have most revenue - BarPlot

plt.figure(figsize = (13,11))

ax = sns.barplot(x = df2["Title"], y = df2["Revenue_millions"], palette = palette)

plt.xlabel('Film Names')

plt.ylabel('Revenue_millions')

plt.title('Top 5 film that have most revenue')
#number of films according to release year

years_list = list(df["Release_year"].unique())

years_number_of_films = []

years_total_revenue = []



for i in years_list:

    dff = df[df['Release_year'] == i]

    countFilms = dff.Title.count()

    years_number_of_films.append(countFilms)

    totalrevenue = dff["Revenue_millions"].sum()

    years_total_revenue.append(totalrevenue)

   

data = pd.DataFrame({"Release_year": years_list, "number_of_films": years_number_of_films, "Total_Revenue": years_total_revenue })

#data

data2 = data.sort_values(by = "Release_year", ascending = True)

data2

#number of films according to release year

# visualization

plt.figure(figsize = (12,8))

sns.pointplot(x = data2['Release_year'], y = data2["number_of_films"], color = "blue", alpha = 0.8)

plt.xticks(rotation = 45)

plt.yticks(rotation = 45)

plt.xlabel('Release_year')

plt.ylabel('Number_of_Films')

plt.title("Number of Films Release Year Basis")
#total revenue release year basis

plt.figure(figsize = (12, 8))

sns.pointplot(x = data2['Release_year'], y = data2["Total_Revenue"], color = "red", alpha = 0.8)

plt.xticks(rotation = 45)

plt.yticks(rotation = 45)

plt.xlabel('Release_year')

plt.ylabel('Total_Revenue')

plt.title("Total Revenue Release Year Basis")
#top 5 films that have the highest runtimes

x2 = df.sort_values(by = "Runtime_minutes", ascending = False).loc[:, ["Title", "Runtime_minutes"]].head()

x2
plt.figure(figsize = (12, 8))

sns.barplot(x = x2.Title, y = x2["Runtime_minutes"], palette = sns.dark_palette("cyan"))

plt.xticks(rotation = 45)

plt.yticks(rotation = 45)

plt.xlabel("Film_Names")

plt.ylabel("Runtime_minutes")

plt.title("top 5 films that have the most runtime")
#all films with runtime

x2 = df.sort_values(by = "Runtime_minutes", ascending = False).loc[:, ["Title", "Runtime_minutes"]]
#what is the average runtime of all films?

#in violin plot, we can see the most repetitive values.

#as we can see from the graph, we can conclude that most of the films have about 110 minutes.



plt.figure(figsize = (8, 8))

sns.violinplot(data = x2, inner = "points", palette = "Set1")
#violin plot alternative

#runtime histogram

df["Runtime_minutes"].plot(kind = 'hist', bins = 20, figsize = (10, 10), color = 'brown')
#we are adding a new column that contains the time intervals of runtime_hour.

#For instance, if a film lasts 1.1 hour, its duration is 1-2 hours.



#calculating runtime_hour

x2["Runtime_hour"] = x2["Runtime_minutes"] / 60



#adding Duration column

x2.loc[x2["Runtime_hour"] < 1, "Duration"] = "0-1"

x2.loc[(x2["Runtime_hour"] >= 1) & (x2["Runtime_hour"] < 2) , "Duration"] = "1-2"

x2.loc[(x2["Runtime_hour"] >= 2) & (x2["Runtime_hour"] < 3) , "Duration"] = "2-3"

x2.loc[x2["Runtime_hour"] >= 3 , "Duration"] = "3+"



#preparing new df for duration-film count

duration_list = list(x2["Duration"].unique())

films = []



for i in duration_list:

    x = x2[x2["Duration"] == i]

    filmsCount = x.Title.count()

    films.append(filmsCount)



newdF = pd.DataFrame({"Duration": duration_list, "Number_of_Films": films})

newdF
#visualization

plt.figure(figsize = (10, 7))

plt.xticks(rotation = 45)

plt.yticks(rotation = 45)

sns.barplot(x = newdF["Duration"], y = newdF["Number_of_Films"], palette = sns.dark_palette("red"))
#the most productive directors

#we want to get information about which director made how many films on release year basis. 



directors_list = list(df.Director.unique())

directors_number_of_films = []



for i in directors_list:

    xx = df[df['Director'] == i]

    countFilms = xx.Title.count()

    directors_number_of_films.append(countFilms)

    

data = pd.DataFrame({"Director": directors_list, "number_of_films": directors_number_of_films})

#data

data2 = data.sort_values(by = "number_of_films", ascending = False).head(10)

data2
#the most productive directors

#visualization

plt.figure(figsize = (14, 11))

ax = sns.barplot(x = data2["Director"], y = data2["number_of_films"], palette = palette)

plt.xticks(rotation = 45)

plt.yticks(rotation = 45)

plt.xlabel("Directors")

plt.ylabel("Number_of_Films")

plt.title("The most productive directors")
#highest ratings

b_ratings = df.loc[:, ["Title", "Rating"]].sort_values(by = "Rating", ascending = False).head(15)



#visualization

plt.figure(figsize = (14, 11))

ax = sns.barplot(x = b_ratings["Title"], y = b_ratings["Rating"], palette = palette)

plt.xticks(rotation = 45)

plt.yticks(rotation = 45)

plt.xlabel("Title")

plt.ylabel("Ratings")

plt.title("The films that have biggest ratings")
#top 10 votes and films

#as we can see, Nolan's films are very successful in attracting audiences.

df.loc[:, ["Title", "Director", "Votes"]].sort_values(by = "Votes", ascending = False).head(10)
#how many films are there based on genre?

#splitting the "genres" column by delimiter

df[["genre1", "genre2", "genre3"]] = df["Genre"].str.split(",", expand = True)

df.loc[:, ["Title", "Director", "genre1", "genre2", "genre3"]].head()
#how many films are there based on genre?

#replacing None to NaN

df.replace(to_replace=[None], value=np.nan, inplace=True)

Genres = df.loc[:, ["Title", "Director", "genre1", "genre2", "genre3"]]

#df["genre2"].fillna(value = np.nan, inplace = True)

#df["genre3"].fillna(value = np.nan, inplace = True)

Genres.head()
#how many films are there based on genre?

#melting the dataFrame based on genres(genre1, genre2, genre3)

x1 = Genres.melt(id_vars = ["Title", "Director"], value_vars = ["genre1", "genre2", "genre3"], value_name = "genre")



#list of unique genres

unique_genres = list(x1["genre"].unique())

genres_number_of_films = []



for i in unique_genres:

    x2 = x1[x1['genre'] == i]

    countFilms = x2.Title.count()

    genres_number_of_films.append(countFilms)

    

data = pd.DataFrame({"Genre": unique_genres, "number_of_films": genres_number_of_films})

data2 = data.sort_values(by = "number_of_films", ascending = False).head(10)



data2
#first graph: pie

labels = data2["Genre"]

sizes = data2["number_of_films"]

fig1, ax1 = plt.subplots(figsize = (12, 10))

ax1.pie(sizes, labels = labels, autopct = '%1.1f%%')

plt.show()
#how many films are there based on genre?

#visualization

#second graph: bar

plt.figure(figsize = (13, 11))

ax = sns.barplot(x = data2["Genre"], y = data2["number_of_films"], palette = sns.dark_palette("seagreen"))

plt.xticks(rotation = 45)

plt.yticks(rotation = 45)

plt.xlabel("Genres")

plt.ylabel("Number_of_Films")

plt.title("Film Count Based on Genres")
#directors based on genres they shoot

x1.pivot_table(index = ["Director", "genre"], aggfunc = {"genre": "count"})
#which actor performed in movies at most?



#splitting the "Actors" column

df[["actor1", "actor2", "actor3", "actor4"]] = df["Actors"].str.split(",", expand = True)

Actors = df.loc[:, ["Title", "Director", "actor1", "actor2", "actor3", "actor4"]]

Actors.head(3)



#melting 

Actors_melt = Actors.melt(id_vars = "Title", value_vars = ["actor1", "actor2", "actor3", "actor4"], value_name = "Actor")



#pivot table

Actors_melt.pivot_table(Actors_melt, index = ["Actor"], aggfunc = {"Title": "count"}).sort_values(by = "Title", ascending = False).head(10)
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt
netflix = pd.read_csv("/kaggle/input/netflix-shows/netflix_titles.csv")

netflix.head()
months = {"January":1,

          "February":2,

          "March":3,

          "April":4,

          "May":5,

          "June":6,

          "July":7,

          "August":8,

          "September":9,

          "October":10,

          "November":11,

          "December":12}



for idx, row in enumerate(netflix["date_added"]):

    try:

        year = row.split()[2]

        month = months[row.split()[0]]

        netflix.loc[idx, "year_added"] = year

        netflix.loc[idx, "month_added"] = month

    except AttributeError: #the row is empty

        pass

    

netflix.head()
netflix_by_year = netflix[["type","year_added","month_added"]]

netflix_by_year.groupby(["type","year_added"]).count()
fig, ax = plt.subplots(figsize=(15,7))

netflix_by_year.groupby(["year_added","type"]).count()["month_added"].unstack().plot.bar(ax=ax)

plt.ylabel("count")

plt.xticks(rotation=45)

plt.title("Number of Movie/TV show added from 2008 to 2020")

plt.show()
netflix_2019 = netflix[netflix["year_added"]=="2019"]

result = netflix_2019[["type","year_added", "month_added"]]

result.groupby(["type","month_added"]).count()
fig, ax = plt.subplots(figsize=(15,7))

result.groupby(["month_added","type"]).count()["year_added"].unstack().plot.bar(ax=ax)

plt.ylabel("count")

ax.set_xticklabels(months.keys())

plt.xticks(rotation=45)

plt.title("Number of Movie/TV show added each month in 2019")

plt.show()
director = []

for row in netflix["director"]:

    if type(row) == str: 

        director.extend([v.strip() for v in row.split(",")])



x = np.array(director)

unique_director = np.unique(x)

print("The number of unique directors:", len(unique_director))

unique_director #unique set of directors
netflix_by_director = pd.DataFrame(index=unique_director)

netflix_by_director["title"] = ""

netflix_by_director["count"] = 0

netflix_by_director
for idx in range(len(netflix)):

    title = netflix.loc[idx, "title"]

    directors = netflix.loc[idx, "director"]

    if type(directors) == str: #director is not empty

        for d in directors.split(","):

            netflix_by_director.loc[d.strip(), "count"] += 1 #update count

            if netflix_by_director.loc[d.strip(), "title"] == "": #update title

                netflix_by_director.loc[d.strip(), "title"] += "{}".format(title)

            else:

                netflix_by_director.loc[d.strip(), "title"] += " / {}".format(title)

                

netflix_by_director                
for row in netflix_by_director["title"]:

    list1 = [v.strip() for v in row.split("/")]

    list2 = list(set(list1))

    if len(list1) != len(list2): #check duplicate values

        print(list1, list2)
#remove duplicate values

dup_list = ["A.R. Murugadoss", "Eduardo Chauvet", "G.J. Echternkamp", "Gajendra Ahire","Miguel Cohan"]

for d in dup_list:

    list1 = [v.strip() for v in netflix_by_director.loc[d,"title"].split("/")]

    list2 = list(set(list1))

    netflix_by_director.loc[d,"title"] = " / ".join(list2) #update title

    netflix_by_director.loc[d,"count"] -= 1 #update count
fig, ax = plt.subplots(figsize=(15,7))

netflix_by_director.sort_values(by="count", ascending=False)["count"][:20].plot.bar()

plt.xlabel("directors")

plt.ylabel("count")

plt.xticks(rotation=45)

plt.title("Number of Movie/TV show on Netflix by Directors")

plt.show()
cast = []

for row in netflix["cast"]:

    if type(row) == str: 

        cast.extend([v.strip() for v in row.split(",")])



x = np.array(cast)

unique_cast = np.unique(x)

print("The number of unique casts:", len(unique_cast))

unique_cast #unique set of casts
netflix_by_cast = pd.DataFrame(index=unique_cast)

netflix_by_cast["title"] = ""

netflix_by_cast["count"] = 0

netflix_by_cast
for idx in range(len(netflix)):

    title = netflix.loc[idx, "title"]

    casts = netflix.loc[idx, "cast"]

    if type(casts) == str: #cast is not empty

        for c in casts.split(","):

            netflix_by_cast.loc[c.strip(), "count"] += 1 #update count

            if netflix_by_cast.loc[c.strip(), "title"] == "": #update title

                netflix_by_cast.loc[c.strip(), "title"] += "{}".format(title)

            else:

                netflix_by_cast.loc[c.strip(), "title"] += " / {}".format(title)

                

netflix_by_cast   
netflix_by_cast.reset_index(inplace=True) #reset index 

netflix_by_cast.head()
#remove duplicate values

for idx in range(len(netflix_by_cast)):

    list1 = [v.strip() for v in netflix_by_cast.loc[idx,"title"].split("/")]

    list2 = list(set(list1))

    if len(list1) != len(list2):

        netflix_by_cast.loc[idx,"count"] = len(list2) #update count

        netflix_by_cast.loc[idx,"title"] = " / ".join(list2) #update title

        

netflix_by_cast.set_index("index", inplace=True)

netflix_by_cast
fig, ax = plt.subplots(figsize=(15,7))

netflix_by_cast.sort_values(by="count", ascending=False)["count"][:20].plot.bar() 

plt.xlabel("casts")

plt.ylabel("count")

plt.xticks(rotation=45)

plt.title("Number of Movie/TV show on Netflix by Casts")

plt.show()
ratings = pd.read_csv("/kaggle/input/netflix-movies-and-tv-shows-ratings/IMDB_results_jan-28-2020.csv")

ratings.head()
for idx in range(len(ratings)):

    ratings.loc[idx, "IMDB_title_name"] = ratings.loc[idx, "IMDB_title_name"].split("(")[0].strip().lower()

    if ratings.loc[idx, "IMDB_rating"] != "Not Found":

        ratings.loc[idx, "IMDB_rating_cnt"] = ratings.loc[idx, "IMDB_rating"].split()[3] #update rating

        ratings.loc[idx, "IMDB_rating"] = ratings.loc[idx, "IMDB_rating"].split()[0] #update rating count

        

ratings
netflix["title_lower"] = netflix["title"].apply(lambda x: x.lower()) 

netflix_with_ratings = netflix.set_index("title_lower").join(ratings.set_index('IMDB_title_name')) #INNER JOIN netflix and IMDB data

netflix_with_ratings = netflix_with_ratings[["title","IMDB_rating","IMDB_rating_cnt","country"]]

netflix_with_ratings.reset_index(inplace=True)

netflix_with_ratings.drop("index", axis=1, inplace=True)

netflix_with_ratings
netflix_with_ratings_dropna = netflix_with_ratings.dropna() #drop rows with NaN values

netflix_with_ratings_dropna.reset_index(inplace=True)

netflix_with_ratings_dropna.drop("index", axis=1, inplace=True)

netflix_with_ratings_dropna
netflix_with_ratings_sorted = netflix_with_ratings_dropna.sort_values(by="IMDB_rating", ascending=False)

netflix_with_ratings_sorted.reset_index(inplace=True)

netflix_with_ratings_sorted.drop("index", axis=1, inplace=True)

netflix_with_ratings_sorted
netflix_with_ratings_sorted["IMDB_rating"] = netflix_with_ratings_sorted["IMDB_rating"].astype(float) #convert data type into float
#check distribution of IMDB ratings

print(netflix_with_ratings_sorted["IMDB_rating"].describe())

fig, ax = plt.subplots(figsize=(15,7))

plt.hist(netflix_with_ratings_sorted["IMDB_rating"])

plt.xlabel("IMDB ratings")

plt.ylabel("count")

plt.title("Distribution of IMDB ratings")

plt.show()
netflix_high_ratings = netflix_with_ratings_sorted[netflix_with_ratings_sorted["IMDB_rating"]>=7.4]

netflix_high_ratings
country = []

for row in netflix_high_ratings["country"]:

    country.extend([v.strip() for v in row.split(",")])



x = np.array(country)

unique_country = np.unique(x)

print("The number of unique countries:", len(unique_country))

unique_country #unique set of countries
netflix_by_country = pd.DataFrame(index=unique_country)

netflix_by_country["title"] = ""

netflix_by_country["count"] = 0
for idx in range(len(netflix_high_ratings)):

    title = netflix_high_ratings.loc[idx, "title"]

    countries = netflix_high_ratings.loc[idx, "country"]

    for c in countries.split(","):

        netflix_by_country.loc[c.strip(), "count"] += 1 #update count

        if netflix_by_country.loc[c.strip(), "title"] == "": #update title

            netflix_by_country.loc[c.strip(), "title"] += "{}".format(title)

        else:

            netflix_by_country.loc[c.strip(), "title"] += " / {}".format(title)

                

netflix_by_country   
netflix_by_country_sorted = netflix_by_country.sort_values(by="count", ascending=False)

netflix_by_country_sorted["ratio"] = round((netflix_by_country_sorted["count"]/netflix_by_country_sorted["count"].sum())*100, 2)

netflix_by_country_sorted
netflix_by_country_top10 = netflix_by_country_sorted

netflix_by_country_top10.loc["Other","count"] = netflix_by_country_sorted.loc["Germany":"Zimbabwe","count"].sum()

netflix_by_country_top10.loc["Other","ratio"] = netflix_by_country_sorted.loc["Germany":"Zimbabwe","ratio"].sum()

netflix_by_country_top10.drop(index=netflix_by_country_top10["Germany":"Zimbabwe"].index, axis=0, inplace=True)

netflix_by_country_top10
fig, ax = plt.subplots(figsize=(15,7))

sizes = netflix_by_country_top10["ratio"]

labels = netflix_by_country_top10.index

theme = plt.get_cmap('hsv') 

ax.set_prop_cycle("color", [theme(1. * i / len(sizes)) for i in range(len(sizes))])

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)

plt.title("Top 10 countries on Netflix with high rating shows")

plt.show()
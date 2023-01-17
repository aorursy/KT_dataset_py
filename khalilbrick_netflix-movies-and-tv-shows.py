#libraries

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline
netflix = pd.read_csv("../input/netflix-shows/netflix_titles.csv")

netflix = netflix.set_index("title")

netflix.head()
movies = netflix.loc[netflix.type == "Movie"]

movies_year_count = movies.release_year.value_counts().sort_values()

shows = netflix.loc[netflix.type == "TV Show"]

shows_year_count = shows.release_year.value_counts().sort_values()

shows_year_count
plt.figure(figsize = (15,8))

plt.bar(movies_year_count.index,movies_year_count.values,label = "Movies")

plt.bar(shows_year_count.index,shows_year_count.values,label = "TV Shows",color = "black")

plt.legend(fontsize = 15)

plt.xticks(size = 15)

plt.yticks(size = 15)

plt.ylabel("Number",size = 15)

plt.title("Movies - TV Show number from 1925 to 2020",fontdict = {"fontsize":25})

plt.savefig("Movies - TV Show number from 1925 to 2020",dpi = 300)

plt.show()
Movies_Country = netflix.loc[netflix.type == "Movie"].country.value_counts().sort_values()[-1:-54:-1].rename("Movies")

Shows_Country = netflix.loc[netflix.type == "TV Show"].country.value_counts().sort_values().rename("TV Shows")

Movies_Shows_Country = pd.merge(Movies_Country,Shows_Country,how = "left",left_index = True,right_index = True)[0:24]

Movies_Shows_Country = Movies_Shows_Country.rename(index = {"United States":"US","United Kingdom":"UK","United Kingdom, United States":"UK, US","United States, Canada":"US, Canada","United States, United Kingdom":"UK, US","Canada, United States":"US, Canada"})

ind_countries = Movies_Shows_Country.index.unique()



Movies_Shows_Country_final_remove_dublicates = pd.DataFrame(index = ind_countries,columns = ["Movies","TV Shows"])

for country in ind_countries:

    data_country = Movies_Shows_Country.loc[country]

    data_country_final_movies = data_country.Movies.sum(axis = 0)

    data_country_final_shows = data_country["TV Shows"].sum(axis = 0)



    Movies_Shows_Country_final_remove_dublicates.loc[country]["Movies"] = data_country_final_movies

    Movies_Shows_Country_final_remove_dublicates.loc[country]["TV Shows"] = data_country_final_shows

Movies_Shows_Country_final_remove_dublicates = Movies_Shows_Country_final_remove_dublicates.sort_values(by = "Movies")

Movies_Shows_Country_final_remove_dublicates = Movies_Shows_Country_final_remove_dublicates.fillna(0)
Movies_Shows_Country_final_remove_dublicates[-1::-1].plot(kind = "bar",figsize = (15,8))

plt.legend(fontsize = 15)

plt.ylabel("Number",size = 15)

plt.yticks(size = 15)

plt.xticks(rotation = 80)

plt.title("Movies - TV Shows by Country",size = 25)

plt.savefig("Movies - TV Shows by Country",dpi = 300)

plt.show()
Directors  = netflix.director.value_counts().sort_values()[-1:-11:-1]

Directors
plt.figure(figsize = (15,8))

plt.bar(Directors.index,Directors.values)

plt.plot(Directors.index,Directors.values,marker = "o",color ="b")



plt.xticks(size = 10,weight = "bold",rotation = 20)



plt.yticks([i for i in range(0,19,2)],size = 15)

plt.title("Directors of Movies and TV Shows",size = 25)

plt.savefig("Directors of Movies and TV Shows",dpi = 300)

plt.show()
MoviesOnStreamingPlatforms_updated = pd.read_csv("../input/netflix-imdb/MoviesOnStreamingPlatforms_updated.csv")

MoviesOnStreamingPlatforms_updated = MoviesOnStreamingPlatforms_updated.drop(["ID","Unnamed: 0"],axis = 1)

MoviesOnStreamingPlatforms_updated = MoviesOnStreamingPlatforms_updated.set_index("Title")

MoviesOnStreamingPlatforms_updated_top_10_IMDb = MoviesOnStreamingPlatforms_updated.sort_values(by = "IMDb").dropna()



MoviesOnStreamingPlatforms_updated_top_10_IMDb = MoviesOnStreamingPlatforms_updated_top_10_IMDb[-1:-11:-1]

MoviesOnStreamingPlatforms_updated_top_10_IMDb
plt.figure(figsize = (15,8))

plt.barh(MoviesOnStreamingPlatforms_updated_top_10_IMDb.IMDb.index,MoviesOnStreamingPlatforms_updated_top_10_IMDb.IMDb.values)



plt.show()
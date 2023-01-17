# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import random as rd

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

netflix_df = pd.DataFrame()

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        netflix_df = pd.read_csv(os.path.join(dirname, filename))

netflix_df
def year_added(x):

    if isinstance(x["date_added"],float):

        return "N/A"

    else:

        date_added = x["date_added"]

        return date_added[-4:]

netflix_df["year_added"] = netflix_df.apply(lambda row: year_added(row), axis = 1)

netflix_df
movie_or_show_by_year = netflix_df.groupby(["year_added", "type"]).count()

# Only keep the date_added column which will reflect the total number for each subgroup

# if it has a date_added entry it will also have a year_added entry

movie_or_show_by_year = movie_or_show_by_year["date_added"]

movie_or_show_by_year
year = 2008

movie_vals = []

show_vals = []

for i in range(12):

    movie_vals.append(movie_or_show_by_year[str(year), "Movie"])

    if movie_or_show_by_year.index.isin([(str(year), "TV Show")]).any():

        show_vals.append(movie_or_show_by_year[str(year), "TV Show"])

    else:

        show_vals.append(0)

    year = year + 1

print(movie_vals)

print(show_vals)
years = np.arange(2008, 2020)

plt.plot(years, movie_vals, "r.-", label = "Movies Added")

plt.plot(years, show_vals, "b.-", label = "Shows Added")

plt.xlabel("Year")

plt.ylabel("Number of content added")

plt.title("Shows vs Movies Added to Netflix each year")

plt.legend()

# Any results you write to the current directory are saved as output.

country_dict = {}

for countries in netflix_df["country"]:

    if isinstance(countries,float):

        continue

    if "," in countries:

        split_countries = countries.split(",")

        for country in split_countries:

            if str.strip(country) not in country_dict:

                country_dict[str.strip(country)] = 1

            else:

                country_dict[str.strip(country)] = country_dict[str.strip(country)] + 1

    else:

        if str.strip(countries) not in country_dict:

                country_dict[str.strip(countries)] = 1

        else:

            country_dict[str.strip(countries)] = country_dict[str.strip(countries)] + 1

country_df = pd.DataFrame.from_dict(country_dict, orient="index", columns=["Num of Occurance"])

country_df = country_df.sort_values(by="Num of Occurance", ascending = True)

plt.figure(figsize = (5, 3), dpi = 100)

plt.barh(country_df.index[-10:], country_df["Num of Occurance"].tail(10), align='center', height = 0.5)

plt.title("Top 10 Most Occuring Countries of Netflix Shows")

plt.ylabel("Country")

plt.xlabel("Number of Occurances")
maturity_dict = {}

for rating in netflix_df["rating"]:

    if rating in maturity_dict:

        maturity_dict[rating] = maturity_dict[rating] + 1

    else:

        maturity_dict[rating] = 1

maturity_df = pd.DataFrame.from_dict(maturity_dict, orient="index", columns=["Num of Occurance"])

maturity_df.sort_values(by="Num of Occurance", ascending = True, inplace = True)

maturity_df.drop([np.nan], inplace = True)

plt.figure(figsize = (5, 3), dpi = 100)

plt.barh(maturity_df.index, maturity_df["Num of Occurance"], align='center', height = 0.5)

plt.title("Occurances of Ratings in Netflix Content")

plt.xlabel("# of Occurances")

plt.ylabel("Rating")
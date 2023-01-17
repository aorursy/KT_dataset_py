# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import plotly.express as px

from matplotlib import pyplot as plt

import datetime as dt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
## import file



video = pd.read_csv ("../input/videogames-sales-dataset/Video_Games_Sales_as_at_22_Dec_2016.csv", encoding="utf-8")



##convert year to datetime



video["Year_of_Release"] = pd.to_datetime(video["Year_of_Release"], format='%Y')





##sort by date for timeseries



video = video.sort_values(by='Year_of_Release') 



video["year"] = video["Year_of_Release"].dt.year



video.head(10)
sales = ["NA_Sales","EU_Sales","JP_Sales","Global_Sales","Other_Sales"]
## make a for loop for efficiency



for i in sales:



    plt.figure(figsize=(15,16))

    ax = sns.barplot(x=i, y="Genre", hue="Genre",

                     data=video,dodge=False)

    plt.title(f"{i} per Genre")

    plt.xlabel('Sales (M)')

    plt.ylabel('Genre')


plt.figure(figsize=(15,7))

ax = sns.barplot(x="year", y="Global_Sales", hue="Genre",

                 data=video, dodge=False)

ax.legend(loc="upper right")

plt.title("Global Sales per Genre")

plt.xlabel('Release Year')

plt.ylabel('Sales (M)')

plt.xticks(rotation='vertical')

plt.yticks(rotation='vertical');
## groupby year and aggregate by sum



year_sum = video.groupby(["year"]).agg({"sum"}).reset_index()
fig, ax = plt.subplots(figsize=(16,9))

ax.plot(year_sum["year"],

        year_sum["NA_Sales"],

        color="r",

        label="North American Sales");

ax.plot(year_sum["year"],

        year_sum["EU_Sales"],

        color="b",

        label="Europe Sales");

ax.plot(year_sum["year"],

        year_sum["JP_Sales"],

        color="m",

        label="Japan Sales");

ax.plot(year_sum["year"],

        year_sum["Other_Sales"],

        color="k",

        label="Other Sales");



ax.set_title("Sales per Release Year")

plt.xlabel('Release_year')

plt.ylabel('Sales (M)')

ax.legend();
def first_j(video):

    return video.sort_values(by='JP_Sales')[-3:]
top_j = video.groupby(['year'], group_keys=False).apply(first_j)[['year', 'Name', 'JP_Sales', 'Genre', 'Platform', 'Publisher']]
top_j
top_j["JP_Sales"].value_counts()
## define function to apply for aggregate application



def first(video):

    return video.sort_values(by='Global_Sales')[-1:]
top = video.groupby(['year'], group_keys=False).apply(first)[['year', 'Name', 'Global_Sales', 'Genre', 'Platform', 'Publisher']]
top
## subset for Nintendo 



n = top[top.Publisher == "Nintendo"]
### calculate percentage of overall sales



s = n["Global_Sales"].sum()

t = video["Global_Sales"].sum()

u = s/t

top["Publisher"].value_counts()
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings as ws

ws.filterwarnings ("ignore")
df = pd.read_csv("/kaggle/input/titled-chess-players-india-july-2020/indian_titled_players_july_2020.csv")
df.head()
df.isna().sum()
# clean column name 

def clean_name (name):

    return  name.strip().lower().replace(" ", "_")



df.rename(columns=clean_name, inplace = True)
# Comparing the ratio of gender

sns.set_style("white")

plt.title("The Indian Chess player ratio")

sns.countplot(df.gender.replace({"M":"Male", "F" : "Female"}))

plt.show()
df.head()
df.shape


temp = df.title.value_counts().reset_index()

plt.title("Title distribution of Chess player")

sns.set()

sns.barplot(x = "index", y = "title", data = temp)

plt.show()
# average age of the chess player



birth_year = df.year_of_birth.values
age_value =  2020 - birth_year
print("Average age is ", round(age_value.mean(), 2), "years")
# Average rating as per title

avg_rating_per_title = round(df.groupby("title")["standard_rating"].mean(), 2).reset_index()
plt.title("The average standered rating of the player as per title")

sns.barplot(x = "title", y="standard_rating", data= avg_rating_per_title)

plt.show()
avg_rapid_rating_per_title = round(df.groupby("title")["rapid_rating"].mean(), 2).reset_index()

plt.title("The average Rapid rating of the player as per title")

sns.barplot(x = "title", y="rapid_rating", data= avg_rapid_rating_per_title)

plt.show()

avg_blitz_rating_per_title = round(df.groupby("title")["blitz_rating"].mean(), 2).reset_index()

plt.title("The average Blitz rating of the player as per title")

sns.barplot(x = "title", y="blitz_rating", data= avg_blitz_rating_per_title)

plt.show()

# Now performing the analysis of the grandmaster
temp = df[df["title"] == "GM"]["gender"]

print("No. of grandmaster in india " , temp.count())
sns.countplot(temp)

plt.show()
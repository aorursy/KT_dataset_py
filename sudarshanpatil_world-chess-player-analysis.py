import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import warnings as ws

ws.filterwarnings("ignore")
df = pd.read_csv("/kaggle/input/world-top-chess-players-august-2020/top_chess_players_aug_2020.csv")
df.head()
df.dtypes
df.isna().sum()
df.shape
def clean_name(name):

    return name.lower().strip().replace(" ", "_")

df.rename(columns = clean_name, inplace = True)
df.year_of_birth.fillna(0, inplace = True)
df.year_of_birth = df.year_of_birth.astype(int)
df.head()
df.drop("inactive_flag", axis = 1, inplace = True)
num_col = list(df.dtypes[df.dtypes=="float64"].index)

df[num_col] = df[num_col].fillna(df.median())
df.isna().sum()
df.title = df.title.fillna('FM')
# Gender Wise representation

plt.figure(figsize = (8,8))

plt.title("Gender wise representation of the chess player", fontdict= {"fontsize": 20})

sns.countplot(df["gender"].replace({'M':"Male", "F" : "Female"}), palette="inferno")

plt.xlabel("Gender")

plt.ylabel("Count of Player")



plt.show()
def top_player(rating_type):

    print("Top 5 Player (according to ", rating_type , ")")

    print("--------------------------------------")

    print(df.sort_values(by =["standard_rating"], ascending = False)["name"][:5].reset_index().drop("index", axis = 1))

    print("--------------------------------------\n \n")
top_player("standard_rating")

top_player("blitz_rating")

top_player("rapid_rating")
df.title.value_counts()
# Gender Wise representation

plt.figure(figsize = (8,8))

sns.set()

plt.title("Title wise representation of the chess player", fontdict= {"fontsize": 20})

sns.countplot(df[df['title'] != "FM"]["title"], palette="inferno")

plt.xlabel("Title")

plt.ylabel("Count of Player")

plt.show()
avg_rating_per_title_per_catagory = df.groupby("title")[["standard_rating", "rapid_rating", "blitz_rating"]].mean().reset_index()

fn = pd.concat([avg_rating_per_title_per_catagory.title, avg_rating_per_title_per_catagory[ avg_rating_per_title_per_catagory.columns.to_list()[1:]].astype(int)], axis = 1)
print(fn)
# Gender Wise representation

plt.figure(figsize = (12,8))

sns.set()

plt.title("Distribution of standard Rating per Title of the chess player", fontdict= {"fontsize": 20})

sns.boxplot(x ="title", y= "standard_rating", data = df, palette="inferno", sym = "")

plt.xlabel("Title")

plt.ylabel("Standerd Rating of Player")

plt.show()
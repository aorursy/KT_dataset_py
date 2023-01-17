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
data = pd.read_csv("/kaggle/input/videogamesales/vgsales.csv")
# first 5 rows of dataset

data.head()
# Shape of dataset

data.shape
# dataset info

data.info()
data["Genre"].value_counts()
# import visualize package

import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style("whitegrid")

plt.subplots(figsize=(14, 6))

sns.countplot(x=data["Genre"], palette="Blues_d", data=data)

plt.title("Frequency of Game Genre", fontweight="bold", fontsize=20)
plt.subplots(figsize=(14, 6))

sns.barplot(x=data["Genre"], y=data["Global_Sales"], data=data)

plt.title("Genre Vs Global Sales", fontweight="bold")
pub_game = data["Publisher"].value_counts()[:30]
pub_game.plot(kind="bar", figsize=(14,6), color="#e35e5e")

plt.title("First 30 company, realse most of the games", fontweight="bold")
# Top 30 company Sells

def zone_sells(groupobj):

    sales = data.groupby([groupobj])["Global_Sales", "NA_Sales" ,"EU_Sales" ,"JP_Sales", "Other_Sales"].sum().sort_values("Global_Sales", ascending=False)[:30]

    sales_ = pd.DataFrame(sales)

    sales_.plot(kind="bar", width = 0.9, figsize=(22,7))

    plt.title(f"{groupobj} selling zone", fontweight="bold")
zone_sells("Publisher")
zone_sells("Genre")
zone_sells("Platform")
plt.subplots(figsize=(14, 6))

sns.countplot(x=data["Year"], palette="Blues_d", data=data)

plt.title("Frequency of Game Release", fontweight="bold", fontsize=20)

plt.xticks(rotation=90)
# Most of the game release in 2008 and 2009
year_sale = data.groupby(['Year'])['Global_Sales'].sum()

year_sale = year_sale.reset_index()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)



ax1.xaxis.set_tick_params(rotation=90)

ax2.xaxis.set_tick_params(rotation=90)



plt.suptitle('Global Sales Of Games (In Million)', fontweight="bold")

sns.barplot(x=year_sale["Year"], y=year_sale["Global_Sales"], palette="Blues_d", ax=ax1)

ax2.plot(year_sale["Year"], year_sale["Global_Sales"], linewidth=3)
plt.subplots(figsize=(10,6))

plt.xticks(rotation=90)

sns.countplot(x=data["Platform"])
# From Year1 to Year2 top 30 games and genre count



def game_rank(*years):

    old_games = data[(data["Year"] >= years[0]) & (data["Year"] <= years[1])].sort_values("Year")

    old_games = pd.DataFrame(old_games)

    

    min_rank = old_games["Rank"].min()

    name = old_games[old_games["Rank"]==min_rank][["Name", "Global_Sales"]]

    

    a = name["Name"].values

    sale = name["Global_Sales"].values

    old_30 = data[(data["Year"] >= years[0]) & (data["Year"] <= years[1])].sort_values("Global_Sales", ascending=False)[:30]

    name_sale = old_30.groupby("Name")["Global_Sales"].sum().sort_values(ascending=False)

    

    print(f"The Year From {years[0]} to {years[1]} Name of the game is: {a}, It's Rank: {min_rank} And It's Global Sales is {sale} million")

    

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    

    sns.countplot(x=old_games["Genre"], ax=ax1, palette="Paired")

    ax1.set_title(f"The Year From {years[0]} to {years[1]} Games Genre Count", fontweight="bold")

    

    name_sale.plot(kind="bar", color=["#ff6347"])

    ax1.xaxis.set_tick_params(rotation=90)

    plt.title(f"The Year From {years[0]} to {years[1]} Games Global Sales of 30 games", fontweight="bold")
game_rank(1980, 1990)
game_rank(1991, 2000)
game_rank(2001, 2010)
game_rank(2011, 2020)
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib as mpl

from matplotlib import pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
video_game_df = pd.read_csv("/kaggle/input/videogamesales/vgsales.csv") #read .csv file

video_game_df = video_game_df.loc[video_game_df.Year <= 2015] #ignore after 2015

video_game_df.head()
#check nan existence

video_game_df.isna().any() # Year and Publisher has nan value but i will ignore these

platform_order = video_game_df.Platform.value_counts().iloc[:15]



#visualization

f, ax = plt.subplots(figsize=(15,10))

g = sns.countplot(data=video_game_df, y="Platform", order=platform_order.index) #order is good option to sort and limit counted values

plt.ylabel("Platforms", fontsize=15, color="gray")

plt.xlabel("Published games", fontsize=15, color="gray")

plt.title("Platforms with the most published games between 1980-2020", fontsize=18)

for i, v in enumerate(platform_order): # add number of games for each platform bar

    g.text(v + 10, i + 0.1, v, color="gray")

plt.show()
#problem is one game can published more than one platform and we will group these games -> e.g GTA V has published in 5 platforms

#check existence -> video_game_df.Name.isin(["Grand Theft Auto V"]).any() -> answer is True

check = video_game_df.Name.isin(["Grand Theft Auto V"])

GTA_V = video_game_df[video_game_df.Name == "Grand Theft Auto V"]

GTA_V.head(10) # 5 platforms



#we should group dataframe by name

grp = video_game_df.groupby(["Name"]).first()

genre_order = grp.Genre.value_counts().index



#visualization

f, ax = plt.subplots(figsize=(15,10))

sns.countplot(data=grp, y="Genre", order=genre_order, palette="GnBu_d")

plt.xlabel("Published games", fontsize=15)

plt.ylabel("Genres", fontsize=15)

plt.title("Genres with the most published games between 1980-2015", fontsize=18)

plt.show()
#check nan value and replace them to "other"

video_game_df.Publisher.isna().any() # answer is true

video_game_df.Publisher.replace([np.nan, "Unknown"], "Others", inplace=True)

grp = video_game_df.groupby(["Name"], as_index=False)

gta = grp.get_group("Grand Theft Auto V")

gta.head(10) # same game has five same publisher in Publisher column

unique_game = grp.first()

order = unique_game.Publisher.value_counts()[:25]



#visualiation

f, ax = plt.subplots(figsize=(15,15))

sns.countplot(data=unique_game, y="Publisher", order=order.index)

plt.xlabel("Published games", fontsize=15)

plt.ylabel("Publishers", fontsize=15)

for c, value in enumerate(order.values):

    ax.text(value, c + 0.1, value, color="gray")

plt.title("Publisher with the most published games between 1980-2015", fontsize=18)

plt.show()
# -> one game has more than one row in dataframe

# -> we should modify video_game_df to make genres and publisher unique too



#find top 1000 by global sales(approximately top %10 of all unique games)

sorted_by_global = video_game_df.sort_values(by="Global_Sales", ascending=False).iloc[:1000] #for platform

unique_sorted_by_global = sorted_by_global.groupby(["Name"]).first() #for genres ans publisher



#plt.pie attributes

platform_sizes = sorted_by_global.Platform.value_counts()[:10].values

genre_sizes = unique_sorted_by_global.Genre.value_counts()[:10].values

publisher_sizes = unique_sorted_by_global.Publisher.value_counts()[:10].values

platform_labels = sorted_by_global.Platform.value_counts()[:10].index

genre_labels = unique_sorted_by_global.Genre.value_counts()[:10].index

publisher_labels = unique_sorted_by_global.Publisher.value_counts()[:10].index

all_explode = [0.05 for i in range(len(platform_sizes))]



#visualization

f, axs = plt.subplots(1, 3, figsize=(20, 10))



axs[0].pie(platform_sizes, explode=all_explode, labels=platform_labels, autopct="%1.1f%%", shadow=False, startangle=90, pctdistance=0.85)

axs[1].pie(genre_sizes, explode=all_explode, labels=genre_labels, autopct="%1.1f%%", shadow=False, startangle=90, pctdistance=0.85)

axs[2].pie(publisher_sizes, explode=all_explode, labels=publisher_labels, autopct="%1.1f%%", shadow=False, startangle=90, pctdistance=0.85)



for ax in axs:

    ax.add_patch(mpl.patches.Circle((0, 0), 0.70, fc="white"))



# Equal aspect ratio ensures that pie is drawn as a circle

axs[0].axis('equal')

axs[1].axis('equal')

axs[2].axis('equal')



#titles

axs[0].set_title("Platform", fontsize=18, color="#487ea1")

axs[1].set_title("Genre", fontsize=18, color="#487ea1")

axs[2].set_title("Publisher", fontsize=18, color="#487ea1")



plt.tight_layout(pad=5.0)

f.suptitle("Platform, genre and publisher distribution of top selling 1000 games", fontsize=21)

plt.show()
#check nan values in year column

#print(video_game_df.Year.isna().any()) # if result is True drop nan values

video_game_df.Year.dropna(inplace=True)

sorted_by_year = video_game_df.Year.value_counts().sort_index()



#visualization

f, ax = plt.subplots(figsize=(20,5))

sns.pointplot(x=[int(m) for m in sorted_by_year.index], y=sorted_by_year.values, linestyles="--", scale=0.8, color="purple")

plt.xticks(rotation=45, fontsize=13)

plt.yticks(fontsize=13)

plt.xlabel("Years", fontsize=16)

plt.ylabel("Published games", fontsize=16)

plt.title("All published games in 1980-2015", fontsize=22)

plt.tight_layout()

plt.grid()

plt.show()

video_game_df.corr()
#get sales columns

corr_df = video_game_df.iloc[:, 6:]



#visualization

f, ax = plt.subplots(figsize=(7, 7))

sns.heatmap(corr_df.corr(), annot=True, linewidths=0.1, linecolor="gray", fmt= '.1f', ax=ax)

plt.xticks(rotation=45)

plt.yticks(rotation=45)

ax.set_title("Correlation between sales", pad=25, fontsize=21)

plt.tight_layout()

plt.show()
from collections import Counter



#get all words

ignore_words = ['The', 'of', 'the', 'no', '&', 'to', 'and', '-', 'in', 'for', 'sales)', 'vs.', 'A', '/', 'de']

games_names_list = []



#ignore pointless words

for member in video_game_df.Name:

    splt = member.split(" ")

    for word in splt:

        if word in ignore_words:

            pass

        else:

            games_names_list.append(word)



#find most commons

with_counter = Counter(games_names_list).most_common(50) 

n, num = zip(*with_counter)



#visualization

f, ax = plt.subplots(figsize=(15, 25))

sns.barplot(y=list(n), x=list(num)) 

for c, value in enumerate(with_counter):

    ax.text(value[1] + 1, c + 0.1, value[1])

plt.title("The 50 most common words in game names", fontsize=20, pad=5)

plt.xlabel("Number of games", fontsize=11, color="#575f6e")

plt.show()
import matplotlib.style as style



#to keep top publishers dataframe

get_unique_publishers = list(video_game_df.Publisher.value_counts().index[:10])

top_publisher = pd.DataFrame(columns=["Genre", "Publisher", "Value"])



for publisher in get_unique_publishers:

    x = video_game_df.loc[video_game_df["Publisher"] == publisher, ["Genre", "Publisher"]]

    y = x.Genre.value_counts()

    temp = pd.DataFrame({"Publisher" : [publisher for i in range(len(y))], "Genre" : list(y.index), "Value" : list(y.values)})

    top_publisher = pd.concat([top_publisher, temp], sort=False)



top_publisher["Value"] = pd.to_numeric(top_publisher["Value"], downcast="float")



#visualization

f, ax = plt.subplots(figsize=(20, 15))

sns.barplot(data=top_publisher, x="Publisher", y="Value", hue="Genre", palette="colorblind")

style.use('seaborn-poster')

style.use('ggplot')

plt.xticks(rotation=60)

plt.ylabel("Number of games")

plt.title("Genre distribution of the 10 most publishing companies", fontsize=20)

plt.show()
#drop nan values

video_game_df.Year.dropna(inplace=True)



unique_genres = list(video_game_df.Genre.value_counts()[:8].index) 

years = video_game_df.Year.unique().astype(int)

new_df = pd.DataFrame(columns=["Genre", "Year", "Number_of_games"])

color_palette = ["#d13328", "#c7bf54", "#79a840", "#42a19f", "#3c3d99", "#9847c4", "#de54c9", "#90bd42"]



#find number of genres all published game between 1980-2015

for c, genre in enumerate(unique_genres):

    for year in years:

        x = video_game_df.loc[(video_game_df["Genre"] == genre) & (video_game_df["Year"] == year), ["Genre", "Year"]]

        temp = pd.Series([genre, year, len(x)], index=new_df.columns)

        new_df = new_df.append(temp, ignore_index=True)



#visualization

f, axs = plt.subplots(8, 1, figsize=(20,40))

for c, genre in enumerate(unique_genres):

    sns.pointplot(data=new_df.loc[new_df.Genre == genre].sort_values(by="Year"), x="Year", y="Number_of_games", ax = axs[c], color=color_palette[c], linestyles="--", scale=0.7)

    axs[c].set_title(genre, fontsize=21, color=color_palette[c])

    axs[c].set_ylabel("Number of games", fontsize = 17.0)

    axs[c].set_xlabel('Year', fontsize = 17.0)

    axs[c].tick_params(axis="x", labelrotation=45)

f.suptitle("Change of Genres between 1980-2015", y=1.01, fontsize=23)

plt.xticks(rotation=45)

plt.tight_layout(h_pad=3.0)

plt.show()
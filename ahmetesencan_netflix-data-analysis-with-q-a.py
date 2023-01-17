import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("../input/netflix-shows/netflix_titles.csv")
df.head()
df.shape
df.info()
df.isnull().sum()
df["country"].value_counts().head()
df["country"].fillna("United States", inplace=True)
df["rating"].value_counts().head()
df["rating"].fillna("TV-MA", inplace=True)
fig, ax = plt.subplots(1,2, figsize=(24,12))

fig.suptitle('Top 5 Countries With Most Contents on Netflix', size=24, fontweight="bold")

df.country.value_counts().head().sort_values().plot(kind = 'barh', color="skyblue",ax=ax[0])

ax[0].set_xlabel("Content Number", fontweight="bold", size=15)

ax[0].xaxis.set_tick_params(labelsize=15)

ax[0].yaxis.set_tick_params(labelsize=15)

ax[0].set_ylabel("Country", fontweight="bold", size=15)



explode = (0, 0, 0, 0 ,0.1)

colors = ["pink", "violet","gray", "green","skyblue"]

labels = df.country.value_counts().head().sort_values().index

values =df.country.value_counts().head().sort_values()

plt.pie(values, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90, textprops={'fontsize': 15, "fontweight" : "bold"}, colors=colors)

fig, ax = plt.subplots(1,2, figsize=(24,12))

fig.suptitle('Distribution of Show Types', size=24, fontweight="bold")

sns.countplot(df["type"], palette = "Blues", ax=ax[0])

ax[0].set_xlabel("Show Type", fontweight="bold", size=15)

ax[0].xaxis.set_tick_params(labelsize=15)

ax[0].yaxis.set_tick_params(labelsize=15)

ax[0].set_ylabel("Content Number", fontweight="bold", size=15)



explode = (0, 0)

colors = ["#83CEAB", "#C97FAB"]

labels = df.type.unique().tolist()

values = df.type.value_counts().tolist()

plt.pie(values, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90, textprops={'fontsize': 15, "fontweight" : "bold"}, colors=colors)
plt.figure(figsize=(16, 8))

sns.countplot(x = 'country',data = df ,hue = 'type', palette = "Purples",

              order = df.country.value_counts().index[0:5])

plt.legend(prop={"size":20}, loc="upper right")

plt.xlabel("Country", size=18)

plt.ylabel("Count", size=18)

plt.title("Comparison of Movies and TV Shows According to Top 5 Countries", size=22, fontweight="bold")

plt.grid()
plt.figure(figsize=(16, 8))

sns.countplot(x='rating',data = df,hue='type', palette = "Oranges")

plt.legend(prop={"size":20}, loc="upper right")

plt.xlabel("Rating", size=18)

plt.ylabel("Count", size=18)

plt.title("Comparison of Movies and TV Shows According to Ratings", size=22, fontweight="bold")

plt.grid()
category = df["listed_in"].str.split(', ', expand=True).stack()

plt.figure(figsize=(16, 8))

category.value_counts().head(25).sort_values(ascending=True).plot(kind="barh", color="#85A18B")

plt.title("Top 25 Categories With Most Contents", size=20)

plt.xlabel("Number of Content", size=15)

plt.ylabel("Category", size=15)
act = df["cast"].str.split(', ', expand=True).stack()

plt.figure(figsize=(16, 8))

act.value_counts().head(10).sort_values(ascending=True).plot(kind="barh", color="#A15878")

plt.title("Top 10 Actors/Actresses With Most Contents", size=20)

plt.xlabel("Number of Content", size=15)

plt.ylabel("Actor/Actress", size=15)
df_nadrop_da =  df.dropna(subset = ["date_added"])
added_year = []

added_month = []

added_day = []



for date in df_nadrop_da.date_added:

    year = (date[-4::1])

    month = date.split()[0]

    day = date.split()[1][0]

    added_year.append(year)

    added_month.append(month)

    added_day.append(day)

    

df_nadrop_da["Added Year"] = added_year

df_nadrop_da["Added Month"] = added_month

df_nadrop_da["Added Day"] = added_day

df_nadrop_da["AY-RY"] = df_nadrop_da["Added Year"].astype(int)- df_nadrop_da["release_year"].astype(int)



df_nadrop_da.head(3)
fig, ax = plt.subplots(1,2, figsize=(24,12))

df_nadrop_da["Added Year"].value_counts().sort_index().plot(kind = 'barh', color="green", ax=ax[0])

fig.suptitle('Variation of Content Number According to Years', size=24, fontweight="bold")

ax[0].set_xlabel("Content Number", size=18)

ax[0].set_ylabel("Year", size=18)

ax[0].xaxis.set_tick_params(labelsize=15)

ax[0].yaxis.set_tick_params(labelsize=15)

x = df_nadrop_da["Added Year"].value_counts().sort_index().index

y = df_nadrop_da["Added Year"].value_counts().sort_index()

ax[1].plot(x, y, linewidth = 4, color="green", marker="o", markerfacecolor="green", markersize=15)

ax[1].set_xlabel("Year", size=18)

ax[1].set_ylabel("Content Number", size=18)

ax[1].xaxis.set_tick_params(labelsize=15)

ax[1].yaxis.set_tick_params(labelsize=15)
plt.figure(figsize=(10,6))

df_nadrop_da["Added Month"].value_counts().plot(kind="bar", color = "gray")

plt.xlabel("Month", size=18, fontweight="bold")

plt.ylabel("Content Number", size=18, fontweight="bold")

plt.xticks(size=15)

plt.yticks(size=15)

plt.title("Variation of Content Number According to Months", size=22, fontweight="bold", fontstyle="italic")
plt.figure(figsize=(10,6))

df_nadrop_da["Added Day"].value_counts().plot(kind="bar", color = "pink")

plt.xlabel("Day", size=18, fontweight="bold")

plt.ylabel("Content Number", size=18, fontweight="bold")

plt.xticks(size=15)

plt.yticks(size=15)

plt.title("Variation of Content Number According to Days of A Month", size=22, fontweight="bold", fontstyle="italic")
df_nadrop_da_mov = df_nadrop_da[df_nadrop_da["type"] == "Movie"]

df_nadrop_da_tv = df_nadrop_da[df_nadrop_da["type"] == "TV Show"]
ave_year_dict = {}

ave_year_dict["Movie"] = df_nadrop_da_mov["AY-RY"].mean()

ave_year_dict["TV Show"] = df_nadrop_da_tv["AY-RY"].mean()



keys = ave_year_dict.keys()

values = ave_year_dict.values()



plt.figure(figsize=(10,6))

plt.bar(keys, values, color = "#BD8787")

plt.xlabel("Type of Show", size=18)

plt.ylabel("Average Year", size=18)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.title("Added Time For Show Types After Being Released", size=20, fontweight="bold", fontstyle="italic")
df_mov = df[df["type"] == "Movie"]

df_tv = df[df["type"] == "TV Show"]



fig, ax = plt.subplots(1,2, figsize=(18,8))

sns.distplot(df_mov["release_year"], ax=ax[0], color="red")

sns.distplot(df_tv["release_year"], ax=ax[1], color="red")

ax[0].set_xlabel("Release Year", size=20)

ax[0].set_title("Distribution of Release Year for Movies", size=25, fontweight="bold", fontstyle="italic")

ax[1].set_xlabel("Release Year", size=20)

ax[1].set_title("Distribution of Release Year for TV Shows", size=25, fontweight="bold", fontstyle="italic")

plt.tight_layout()
df_mov.head(1)
arr_duration_list_mov = []

arr_duration_list_tv = []



for duration in df_mov.duration:

    arr_duration = duration.split()[0]

    arr_duration_list_mov.append(arr_duration)

df_mov["Arranged Duration"] = arr_duration_list_mov

df_mov["Arranged Duration"] = df_mov["Arranged Duration"].astype(int)



for duration in df_tv.duration:

    arr_duration = duration.split()[0]

    arr_duration_list_tv.append(arr_duration)

df_tv["Arranged Duration"] = arr_duration_list_tv

df_tv["Arranged Duration"] = df_tv["Arranged Duration"].astype(int)
fig, ax = plt.subplots(1,2, figsize=(18,8))

sns.distplot(df_mov["Arranged Duration"], ax=ax[0], color="green")

sns.distplot(df_tv["Arranged Duration"], ax=ax[1], color="green")

ax[0].set_xlabel("Duration (min)", size=20)

ax[0].set_title("Distribution of Duration for Movies", size=25, fontweight="bold", fontstyle="italic")

ax[1].set_xlabel("Duration (Season)", size=20)

ax[1].set_title("Distribution of Duration for TV Shows", size=25, fontweight="bold", fontstyle="italic")

plt.tight_layout()
mean_duration_movies = df_mov["Arranged Duration"].mean()

mean_duration_tv_shows = df_tv["Arranged Duration"].mean()



print("Average Duration for Movies is", round(mean_duration_movies, 2), "minutes.")

print("Average Duration for TV Show is", round(mean_duration_tv_shows, 2), "Season(s).")



fig, ax = plt.subplots(1,2, figsize=(16,8))

ax[0].bar(" ", mean_duration_movies, color="skyblue")

ax[1].bar(" ", mean_duration_tv_shows, color="skyblue")

ax[0].set_xlabel("Movie", size=20)

ax[0].set_ylabel("Average Min.", size=20)

ax[0].set_title("Average Duration for Movies", size=25, fontweight="bold", fontstyle="italic")

ax[1].set_xlabel("TV Shows", size=20)

ax[1].set_ylabel("Average Season", size=20)

ax[1].set_title("Average Duration for TV Shows", size=25, fontweight="bold", fontstyle="italic")
longest_movie = df_mov.loc[df_mov["Arranged Duration"] == df_mov["Arranged Duration"].max()]

print("The longest movie is", longest_movie.reset_index()["title"][0], "with",

       longest_movie.reset_index()["duration"][0]+".")

longest_movie
longest_tv_show = df_tv.loc[df_tv["Arranged Duration"] == df_tv["Arranged Duration"].max()]

print("There are", len(longest_tv_show), "longest TV Shows.")

for i in range(len(longest_tv_show)):

    print(str(i+1) + ".TV Show is", longest_tv_show.reset_index()["title"][i], 

          "with", longest_tv_show.reset_index()["duration"][i]+"." )

longest_tv_show
df_mov_old = df_mov.loc[df_mov["release_year"] == df_mov["release_year"].min()]

print("There are", len(df_mov_old), "oldest movies.")

for i in range(len(df_mov_old)):

    print(str(i+1) + ".Movie is", df_mov_old.reset_index()["title"][i], "with release year of", 

          df_mov_old.reset_index()["release_year"][i])

df_mov_old
df_tv_old = df_tv.loc[df_tv["release_year"] == df_tv["release_year"].min()]

print("The oldest TV show is", df_tv_old.reset_index()["title"][0], "with release year of",

       df_tv_old.reset_index()["release_year"][0])

df_tv_old
plt.figure(figsize=(10,6))

df_mov.director.value_counts().head().sort_values().plot(kind="barh", color = "#D77373")

plt.xticks(np.arange(0,21,1))

plt.xlabel("Number of Movies", size=15)

plt.ylabel("Director", size=15)

plt.title("Top 5 Directors With Most Movies", size=22, fontweight="bold", fontstyle="italic")
plt.figure(figsize=(10,6))

df_tv.director.value_counts().head().sort_values().plot(kind="barh", color = "#8DB7B9")

plt.xticks(np.arange(0,6,1))

plt.xlabel("Number of TV Shows", size=15)

plt.ylabel("Director", size=15)

plt.title("Top 5 Directors With Most TV Shows", size=22, fontweight="bold", fontstyle="italic")
df_comedy_mov = df_mov[df_mov.listed_in.str.contains('Comedy') | df_mov.listed_in.str.contains("Comedies")] 

df_comedy_tv = df_tv[df_tv.listed_in.str.contains('Comedy') | df_tv.listed_in.str.contains("Comedies")] 



df_horror_mov = df_mov[df_mov.listed_in.str.contains('Horror') | df_mov.listed_in.str.contains("horror")] 

df_horror_tv = df_tv[df_tv.listed_in.str.contains('Horror') | df_tv.listed_in.str.contains("horror")] 
labels = ["Movie", "TV Show"]

comedy_count = [df_comedy_mov.shape[0], df_comedy_tv.shape[0]]

horror_count = [df_horror_mov.shape[0], df_horror_tv.shape[0]]

x = np.arange(len(labels))

width = 0.40



fig, ax = plt.subplots(figsize=(16,8))

rects1 = ax.bar(x - width/2, comedy_count, width, label='Comedy', color="#9DB0B0")

rects2 = ax.bar(x + width/2, horror_count, width, label='Horror', color="#979870")



ax.set_xlabel('Show Type', fontsize=15)

ax.set_ylabel('Content Number', fontsize=15)

ax.set_title('Comparison of Comedy and Horror Contents', fontsize=22, fontweight="bold", fontstyle="italic")

ax.set_xticks(x)

ax.set_xticklabels(labels)

plt.xticks(fontsize=16)

legend = ax.legend(loc='upper right',prop={"size":18})

legend.set_title('Category',prop={'size':20})
df_tv_trans = df_tv[df_tv.title.str.contains('Transformers') | df_tv.title.str.contains("transformers")]



print("There are", df_tv_trans.shape[0], "Transformers shows.")

df_tv_trans
df_tv_trans.sort_values(by=['release_year'], inplace=True, ascending=False)



print("The newest Transformers show on Netflix is", df_tv_trans.head(1).reset_index()["title"][0], 

      "with release year of", df_tv_trans.head(1).reset_index()["release_year"][0])

df_tv_trans.head(1)
df_war = df[df.description.str.contains('War') | df.description.str.contains("war")]

print("There are", df_war.shape[0], 'contents which contain the word "war".')
plt.figure(figsize=(10,6))

df_war["country"].value_counts().head().plot(kind="bar", color = "#B5C338")

plt.title('Countries Which Have The Most Contents That\n Contain The Word "war" In The Description', size=22, 

          fontweight="bold", fontstyle="italic")

plt.xlabel("Country", size=18)

plt.ylabel("Content Number", size=18)

plt.xticks(size=15)

plt.yticks(size=15)
df_love = df[df.description.str.contains('Love') | df.description.str.contains("love")]

print("There are", df_love.shape[0], 'contents which contain the word "love".')
plt.figure(figsize=(10,6))

df_love["country"].value_counts().head().plot(kind="bar", color="#A92E43")

plt.title('Countries Which Have The Most Contents That\n Contain The Word "love" In The Description', size=22, 

          fontweight="bold", fontstyle="italic")

plt.xlabel("Country", size=18)

plt.ylabel("Content Number", size=18)

plt.xticks(size=15)

plt.yticks(size=15)
df_will_eva = df[df.cast.str.contains('Will Smith') & df.cast.str.contains("Eva Mendes")]



print("The", df_will_eva.reset_index()["type"][0] + ' "' +df_will_eva.reset_index()["title"][0]+ '" ' +

     "has both Will Smith and Eva Mendes in its cast.")

df_will_eva
df_burton = df[df.director.str.contains('tim burton') | df.director.str.contains("Tim Burton")]

df_burton = df_burton[df_burton["duration"] == df_burton["duration"].max()]



print("The longest movie directed by Tim Burton is" + ' "'+df_burton.reset_index()["title"][0]+'" ' 

      "with", df_burton["duration"].max() + ".")

df_burton
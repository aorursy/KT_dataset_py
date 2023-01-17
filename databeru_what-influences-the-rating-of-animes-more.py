import pandas as pd

import numpy as np

df = pd.read_csv('../input/crunchyroll-anime-ratings/animes.csv')
df.describe()
# Change the type of the columns with the genre to integer

for c in df.columns[12:]:

    df[c] = df[c].astype("int")

    

# Delete "genre_" from the columns to have clean genres

idx = []

for i in df.columns:

    idx.append(i.replace("genre_",""))

df.columns = idx
import seaborn as sns

import matplotlib.pyplot as plt

sns.heatmap(df.isnull())

plt.title("Missing values?", fontsize = 18)

plt.show()
# Get the number of animes with how many genres

number_genre = df[df.columns[12:]].sum(axis = 1).value_counts().sort_index()



# Create a color list depending on the number the genres

colors = []

maxg = max(number_genre)

for n in number_genre:

    x = 0.8 - n/(2*maxg)

    colors.append((0.7, x, x))

average = number_genre.mean()



# Plot the result

plt.figure(figsize=(10,6))

number_genre.plot.bar(color = colors)

plt.title("Repartition of the number of genres", fontsize = 18)

plt.axhline(average, 0 ,1, color = "black", lw = 3)

plt.text(9.6, 120, "average", fontsize = 14)

plt.ylabel("Animes count", fontsize = 14)

plt.xlabel("\nNumber of genres", fontsize = 14)

plt.show()
def transf(x):

# Some animes don't have any category, 

# to avoid diving by 0, categories of animes with

# no categories will be divided by 1 instead

    if x == 0:

        return 1

    else:

        return x



def weight_df(df, col_start = 12):

# Transform the genres into weighted genres

    

    fact = df[df.columns[col_start:]].sum(axis = 1).apply(lambda x:transf(x))

    df_va = df.values

    for m in range(len(df_va)):

        df_va[m]

        for i in range(col_start, len(df_va[m])):

            df_va[m][i] = df_va[m][i] / fact[m]

    return pd.DataFrame(df_va, columns=df.columns)

    



lst = [["anime 1", 1,1,0,1,1,0],["anime 2", 0,0,0,0,0,1],["anime 3", 1,0,1,1,0,0]]

cols = ["Anime", "category_1", "category_2", "category_3", "category_4", "category_5", "category_6"]



# Without transformation

example = pd.DataFrame(lst, columns = cols)

example
# After transformation

weight_df(example, col_start = 1)
# Get the weighted categories

df_weighted = weight_df(df)



# Get the number of animes with no genre

nb_0_genre = (df[df.columns[12:]].sum(axis = 1) == 0).sum()



# Calculate the cantity of each category without "no genre"

weighted_betw = df_weighted[df_weighted.columns[12:]].sum()



# Add "no genre"

weighted_betw["NO genre"] = nb_0_genre



# Compute the percentage of each genre

distrib_genre = 100 * weighted_betw/weighted_betw.sum()



# Sort the values

distrib_genre = distrib_genre.sort_values(ascending = False)



# Display the results

plt.figure(figsize =(15,10))

bar = sns.barplot(distrib_genre.index, distrib_genre)

plt.title("Distribution of genres", fontsize = 18)

plt.ylabel("%", fontsize = 18)

bar.tick_params(labelsize=16)



# Rotate the x-labels

for item in bar.get_xticklabels():

    item.set_rotation(90)
mean_ratings = []

for g in df_weighted.columns[12:]:

    rating = ((df_weighted["rate"] * df_weighted[g]).sum()) / df_weighted[g].sum()

    mean_ratings.append([g, rating])



mean_ratings = pd.DataFrame(mean_ratings, columns = ["Genre", "Rating"]).sort_values(by = "Rating", ascending = False)



# Display the results

plt.figure(figsize =(15,10))

bar = sns.barplot("Genre", "Rating", data = mean_ratings, palette = "coolwarm")

plt.title("Mean Rating for each Genre", fontsize = 18)

plt.ylabel("Mean Rating", fontsize = 18)

plt.xlabel("")

bar.tick_params(labelsize=16)



# Rotate the x-labels

for item in bar.get_xticklabels():

    item.set_rotation(90)
# Categorize the number of votes for each anime in 6 bins

def create_bins(v):

    if v > 10000:

        return ">10000"

    elif v > 2000:

        return "2000-10000"

    elif v > 500:

        return "500-2000"

    elif v > 100:

        return "100-500"

    elif v >= 10:

        return "10-100"

    else:

        return "<10"



df["votes_cat"] = df["votes"].apply(create_bins)
plt.figure(figsize=(10,7))

bar = sns.countplot(df["votes_cat"])

plt.ylabel("Count of animes", fontsize = 14)

plt.xlabel("\nNumber of votes", fontsize = 14)

plt.title("Number of animes by number of votes", fontsize = 18)

bar.tick_params(labelsize=14)

plt.show()
rate_votes_cat = pd.pivot_table(df, values = "rate", index = "votes_cat").sort_values(by = "rate", ascending = False)



bar = rate_votes_cat.plot.bar(figsize=(10,7), color = "grey")

plt.title("Mean Rating by number of votes", fontsize = 18)

plt.ylabel("Rating", fontsize = 14)

plt.xlabel("\nNumber of votes", fontsize = 14)

bar.tick_params(labelsize=12)

plt.show()
# Calculate the number of episodes depending on the genre

mean_episodes = []

for g in df_weighted.columns[12:]:

    episodes = ((df_weighted["episodes"] * df_weighted[g]).sum()) / df_weighted[g].sum()

    mean_episodes.append([g, episodes])

mean_episodes = pd.DataFrame(mean_episodes, columns = ["Genre", "episodes"]).sort_values(by = "episodes", ascending = False)



# Display the results

plt.figure(figsize =(15,10))

bar = sns.barplot("Genre", "episodes", data = mean_episodes, palette = "coolwarm")

plt.title("Mean number of episodes for each genre", fontsize = 18)

plt.ylabel("Mean number of episodes", fontsize = 12)

plt.xlabel("")

bar.tick_params(labelsize=16)



# Rotate the x-labels

for item in bar.get_xticklabels():

    item.set_rotation(90)
nb_0_epi = df[df["episodes"] <1].shape[0]

print(f"Number of animes with 0 episode: {nb_0_epi}")

print(f"Total number of animes in the dataset: {df.shape[0]}")
bar = df[df["votes_cat"] == ">10000"].sort_values("rate",ascending = False)[:10].plot.barh("anime", "rate", figsize = (10,10), color = sorted(colors, reverse = True))

bar.tick_params(labelsize=16)

plt.title("Best rated animes with > 10.000 votes", fontsize = 18)

plt.xlabel("Mean rating", fontsize = 14)

plt.ylabel("")

plt.show()  
bar = df[df["votes_cat"] == "2000-10000"].sort_values("rate",ascending = False)[:10].plot.barh("anime", "rate", figsize = (10,10), color = sorted(colors, reverse = True))

bar.tick_params(labelsize=16)

plt.title("Best rated animes with 2.000 - 10.000 votes", fontsize = 18)

plt.xlabel("Mean rating", fontsize = 14)

plt.ylabel("")

plt.show()  
bar = df[df["votes_cat"] == "500-2000"].sort_values("rate",ascending = False)[:10].plot.barh("anime", "rate", figsize = (10,10), color = sorted(colors, reverse = True))

bar.tick_params(labelsize=16)

plt.title("Best rated animes with 500 - 2.000 votes", fontsize = 18)

plt.xlabel("Mean rating", fontsize = 14)

plt.ylabel("")

plt.show()  
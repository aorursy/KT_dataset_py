import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from scipy.stats import mannwhitneyu

from scipy.stats import normaltest



data = pd.read_csv("../input/reviews.csv", index_col=0)
data.head()
print('total number of reviews: ', len(data.index))
genres = list(data.columns[:-2])

print('Among the music genres are: ',end='')

for genre in genres[:7]:

    print(' '+genre+', ',end='')

print('...')



print('total number of genres: ',len(genres))

authors = list(data.AUTHOR.unique())

print('Review authors are denoted by letters: ',authors)
grades = [i for i in range(11)]

print('possible grades range from ',0,' (worst) to ', 10, ' (best) with "-" denoting a missing value')
genre_abs = [data.loc[:,g].sum() for g in genres]

gc_data = pd.DataFrame({'genre' : genres, 'abs_freq': genre_abs}).sort_values('abs_freq',ascending=False)

gc_data = gc_data.iloc[:20,:]



# plot the top ten genres

plt.figure(figsize=(18,10))

plot = sns.barplot(x=gc_data.genre, y=gc_data.abs_freq)

plt.title("Top 20 most reviewed heavy metal subgenres")

plt.xlabel("Heavy metal subgenres")

plt.ylabel("Number of reviews")

t = plt.xticks(rotation=90)

genre_corr = data.loc[:, gc_data.genre.tolist()].corr()

plt.figure(figsize=(16,12))

plot = sns.heatmap(data=genre_corr)
print('There are ',len(data[data.RATING == '-']),' reviews with missing rating. We will exclude them from the dataset.')
data = data[data.RATING != '-']

data.RATING = data.RATING.astype(int)
grade_counts = [len(data[data.RATING == grade]) for grade in grades]

plt.figure(figsize=(14,10))

plot = sns.barplot(x=grades,y=grade_counts)

a = plt.title("Distribution of grades")

a = plt.xlabel("Grades")

a = plt.ylabel("Absolute frequency")

print('The median is ',data.RATING.median())

print('The mean is ', data.RATING.mean())

print('The skew is ',data.RATING.skew())
stat, p_value = normaltest(data.RATING)



if p_value < 0.05:

    print("We reject the null-hypothesis with p_value ", p_value)

else:

    print("We confirm the null_hypothesis with p_value ", p_value)
# returns the favourite genre of an author aut

def favourite(aut):

    rev = data.loc[data.AUTHOR == aut]

    genre_abs = pd.DataFrame({"genre": genres, "counts" : [rev.loc[:,g].sum() for g in genres]})

    return genre_abs.genre[genre_abs.counts.idxmax()]

favourite_genres = pd.DataFrame({"author" : authors, "favourite_genre" : [favourite(aut) for aut in authors]})

favourite_genres
favourite_ratings = pd.Series([])

non_favourite_ratings = pd.Series([])

for aut in authors:

    fav_gen = favourite(aut)

    favourite_ratings = favourite_ratings.append(data[(data.loc[:,fav_gen] == 1) & (data.loc[:,"AUTHOR"] == aut)].RATING)

    non_favourite_ratings = non_favourite_ratings.append(data[(data.loc[:,fav_gen] != 1) & (data.loc[:,"AUTHOR"] == aut)].RATING)

    

print('number of "favourite reviews": ',len(favourite_ratings))

print('number of "non-favourite reviews": ',len(non_favourite_ratings))



fav_cnts = pd.Series([sum(favourite_ratings == i) for i in range(11)])

non_fav_cnts = pd.Series([sum(non_favourite_ratings == i) for i in range(11)])



plt.figure(figsize=(14,10))

a = sns.distplot(favourite_ratings, bins=[0,1,2,3,4,5,6,7,8,9,10,11], hist=True, kde=False, norm_hist=True, label="favourite")

a = sns.distplot(non_favourite_ratings, bins=[0,1,2,3,4,5,6,7,8,9,10,11], hist=True, kde=False, norm_hist=True, label="non-favourite")

plt.title("relative frequency of ratings when reviewing favourite vs. non-favourite genres")

a = plt.legend()
stat, p_value = mannwhitneyu(non_favourite_ratings,favourite_ratings, alternative="less")



if p_value < 0.05:

    print("We reject the null-hypothesis with p_value ",p_value)

else:

    print("We confirm the null-hypothesis with p_value ", p_value)

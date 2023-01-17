import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



tmdb = pd.read_csv("../input/tmdb-movie-metadata/tmdb_5000_movies.csv")

movies = pd.read_csv("../input/movielens-100k-small-dataset/movies.csv")

ratings = pd.read_csv("../input/movielens-100k-small-dataset/ratings.csv")
# defining a function for plot ours series and not repeat code a lot...

def plot_serie(serie):

    f, axes = plt.subplots(2, 2, figsize=(25, 10)) 



    # Axis 1

    ax1 = axes[0,0]

    sns.distplot(serie, norm_hist = False, kde=False, ax=ax1)

    ax1.set(xlabel="Average rate", ylabel="Frequency")

    ax1.set_title("Distribuition of average vote (not normalized)")



    # Axis 2

    ax2 = axes[0,1]

    sns.distplot(serie, ax=ax2)

    ax2.set(xlabel="Average rate", ylabel="Density")

    ax2.set_title("Distribuition of average vote (normalized)")



    # Axis 3

    ax3 = axes[1,0]

    sns.boxplot(x=serie, ax=ax3)

    ax3.set(title='Distribution of average vote', xlabel='')



    # Axis 4

    ax4 = axes[1,1]

    sns.distplot(serie, hist_kws = {'cumulative':True}, kde_kws = {'cumulative':True}, ax=ax4)

    ax4.set(xlabel='Average vote', ylabel='Movies %')

    ax4.set_title('Cumulative average vote')



    f.show()
tmdb.head(2)
# Plot brute date

plot_serie(tmdb.vote_average)

tmdb.vote_average.describe()
tmdb_fmt = tmdb.query('vote_count >= 100')

# Plot formatted date

plot_serie(tmdb_fmt.vote_average)

print(tmdb_fmt.vote_average.describe())
# The TMBD ratings vary from 0 to 10, and here in movie lens, the ratings vary from 0 to 5... we have to fix this, how? just normalize the data. (x 2)

ratings['rating_norm'] = ratings.rating * 2
# Plot brute date

plot_serie(ratings.rating_norm)

print(ratings.rating_norm.describe())
counter_votes = ratings.groupby("movieId").count()

movie_id = counter_votes.query("rating >= 10").index

movies_fmt = ratings.loc[movie_id.values].dropna()

movies_fmt.head(2)
# Plot brute date

plot_serie(movies_fmt.rating_norm)

print(movies_fmt.rating_norm.describe())
from statsmodels.stats.weightstats import DescrStatsW



def print_ci_with_z(serie,name):

    ci = DescrStatsW(serie).zconfint_mean()

    ci_range = round(ci[1] - ci[0],3)

    ci_rounded = (round(ci[0],3),round(ci[1],3))

    string = "The confidence interval (Z) for the {} is {} with a range of {}".format(name, ci_rounded, ci_range)

    print(string) 



def print_ci_with_t(serie,name):

    ci = DescrStatsW(serie).tconfint_mean()

    ci_range = round(ci[1] - ci[0],3)

    ci_rounded = (round(ci[0],3),round(ci[1],3))

    string = "The confidence interval (T) for the {} is {} with a range of {}".format(name, ci_rounded, ci_range)

    print(string)  

    

def print_comp_ci_with_t(serie,name):

    ci = DescrStatsW(serie).tconfint_mean()

    ci_range = round(ci[1] - ci[0],3)

    ci_rounded = (round(ci[0],3),round(ci[1],3))

    string = "The confidence interval (T) for the {} is {} with a range of {}".format(name, ci_rounded, ci_range)

    descr_todas_as_notas = DescrStatsW(notas.rating)

    descr_toystory = DescrStatsW(notas1.rating)

    comparacao = descr_todas_as_notas.get_compare(descr_toystory)

    print(string)  
# Distribution Z

print_ci_with_z(tmdb.vote_average,'brute data')

print_ci_with_z(tmdb_fmt.vote_average,'formatted data')

# Distribution T student

print_ci_with_t(tmdb.vote_average,'brute data')

print_ci_with_t(tmdb_fmt.vote_average,'formatted data')
# Distribution Z

print_ci_with_z(ratings.rating_norm,'brute data')

print_ci_with_z(movies_fmt.rating_norm,'formatted data')

# Distribution T student

print_ci_with_t(ratings.rating_norm,'brute data')

print_ci_with_t(movies_fmt.rating_norm,'formatted data')
from statsmodels.stats.weightstats import ztest,zconfint

# example

ztest(ratings.rating_norm, value = 7.003113967233924)
import numpy as np

import math as mt



def plot_pvalue_confint(series):

    zvalues = zconfint(series)

    nmin = mt.floor(zvalues[0])

    nmax = mt.ceil(zvalues[1])

    nrange = np.arange(nmin,nmax,1/100000)



    pvalues = list()

    for n in nrange:

        aux = ztest(series, value = n)

        if aux[1] != 0.0:

            pvalues.append((n,aux[1]))

    pvalues = pd.DataFrame(pvalues, columns= ['guess','pvalue'])

    f, ax = plt.subplots(figsize=(20, 5))

    sns.scatterplot(x="guess", y="pvalue",

                    linewidth=0,

                    data=pvalues, ax=ax)
plot_pvalue_confint(ratings.rating_norm)
f, ax = plt.subplots(figsize=(10, 5))

sns.violinplot(

    x="movieId",

    y="rating",

    split=True,

    data=ratings[ratings.movieId.isin([2571,45,356])],

    ax=ax

)
# Movie id 2571

movie_2571 = ratings[ratings.movieId == 2571]

desc_movie_2571 = DescrStatsW(movie_2571.rating)



# Movie id 45

movie_45 = ratings[ratings.movieId == 45]

desc_movie_45 = DescrStatsW(movie_45.rating)



# Movie id 356

movie_356 = ratings[ratings.movieId == 356]

desc_movie_356 = DescrStatsW(movie_356.rating)

desc_movie_2571.get_compare(desc_movie_45).summary()
desc_movie_2571.get_compare(desc_movie_356).summary()
desc_movie_45.get_compare(desc_movie_356).summary()
movies["genres"] = movies["genres"].str.replace("-", "")

genres = movies["genres"].str.get_dummies()

print("How much genres we have? %d genres" % genres.shape[1])

gen_col = list(genres.columns)

movies = movies.join(genres).drop(columns=["genres"], axis=1)

movies.head()
def get_rating_by_user(userId):

    return ratings[ratings["userId"] == userId]



def get_rating_by_gen(gen, user_rate):

    ids = movies[movies[gen] == 1].index

    return user_rate[user_rate.movieId.isin(ids)]



def get_rating_user_by_gen(userId):

    user_rating = get_rating_by_user(userId)

    return user_rating.merge(movies, on="movieId")



def best_gen(userId):

    try:

        user_rate = get_rating_user_by_gen(userId)

        rate_by_gen = [get_rating_by_gen(gen, user_rate) for gen in gen_col]

        rate_by_gen = pd.DataFrame(

            [ng.describe()["rating"] for ng in rate_by_gen], index=gen_col

        )

        rate_by_gen["cv"] = (rate_by_gen["std"]) / (rate_by_gen["mean"])

        table = round(

            rate_by_gen[["count", "mean", "std", "cv"]]

            .query("count > 10")

            .sort_values("cv"),

            3,

        )

        return table.index[0]

    except:

       # Caso a pessoa não tenha mais de 10 votos em um especifico genero....

       return "Not found"
%%time

from joblib import Parallel, delayed

    

gen_fav = Parallel(verbose=1, n_jobs=-1)(delayed(best_gen)(uid) for uid in ratings.userId.unique())
users = pd.DataFrame(gen_fav, index=ratings.userId.unique(),columns=['Favorite Genre'])
cout_users = pd.DataFrame(users["Favorite Genre"].value_counts())

cout_users["Genre"] = cout_users.index

plt.figure(figsize=(20, 6))

ax = sns.barplot(x="Genre", y="Favorite Genre", data=cout_users)

plt.title("Number of users by favorite genre")

plt.ylabel("Users")

plt.xlabel("Genre")

plt.show()
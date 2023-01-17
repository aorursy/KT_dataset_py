%matplotlib inline

import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style('ticks')

sns.set_palette('bright')

sns.set_context('notebook', font_scale=1.2)



pd.options.plotting.backend = "plotly" # Setting Plotly as Backend in case I later want to generate an interactive plot. Seaborn will be default for static plots.
animation = pd.read_csv('../input/letterboxd-movies/Animation.csv')

horror = pd.read_csv('../input/letterboxd-movies/Horror.csv')

scifi = pd.read_csv('../input/letterboxd-movies/SciFi.csv')

thriller = pd.read_csv('../input/letterboxd-movies/Thriller.csv')

animation.info()
animation.describe(include='all')
horror.info()
horror.describe(include='all')
scifi.info()
scifi.describe(include='all')
thriller.info()
thriller.describe(include='all')
animation.dropna(axis=0, inplace=True)

horror.dropna(axis=0, inplace=True)

scifi.dropna(axis=0, inplace=True)

thriller.dropna(axis=0, inplace=True)
animation = animation[(50 <= animation.running_time) & (animation.running_time <= 240)]
animation.describe(include='all')
# Remove Over the Garden Wall and FLCL cause they are series.



animation.drop([25, 90], axis=0, inplace=True)
horror = horror[(50 <= horror.running_time) & (horror.running_time <= 240)]
horror.describe(include='all')
scifi = scifi[(50 <= scifi.running_time) & (scifi.running_time <= 240)]
scifi.describe(include='all')
thriller = thriller[(50 <= thriller.running_time) & (thriller.running_time <= 240) & (thriller.year < 2021)]



#Removing 2021 (to be released) from thriller
thriller.describe(include='all')
animation['year'] = animation['year'].astype(int)

horror['year'] = horror['year'].astype(int)

scifi['year'] = scifi['year'].astype(int)

thriller['year'] = thriller['year'].astype(int)
#Extracting descriptive statistics for each genre's avg_rating distribution

animation_rating_mean = round(animation['avg_rating'].mean(), 2)

animation_rating_std = round(animation['avg_rating'].std(), 2)

animation_rating_skp = round(animation['avg_rating'].skew(), 2)



horror_rating_mean = round(horror['avg_rating'].mean(), 2)

horror_rating_std = round(horror['avg_rating'].std(), 2)

horror_rating_skp = round(horror['avg_rating'].skew(), 2)



scifi_rating_mean = round(scifi['avg_rating'].mean(), 2)

scifi_rating_std = round(scifi['avg_rating'].std(), 2)

scifi_rating_skp = round(scifi['avg_rating'].skew(), 2)



thriller_rating_mean = round(thriller['avg_rating'].mean(), 2)

thriller_rating_std = round(thriller['avg_rating'].std(), 2)

thriller_rating_skp = round(thriller['avg_rating'].skew(), 2)
#total movies

animation_n_movies = len(animation)

horror_n_movies = len(horror)

scifi_n_movies = len(scifi)

thriller_n_movies = len(thriller)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10), squeeze=True)

fig.suptitle('Average Rating Distributions per Genre', fontsize=25)



sns.distplot(animation.avg_rating, ax=axes[0,0], norm_hist=True,

             color='y', axlabel=False, label=f'Total of {animation_n_movies} Movies')

axes[0,0].set_title(f'Animation (μ = {animation_rating_mean}, σ = {animation_rating_std}, Skp = {animation_rating_skp})',

                    fontsize=20)

axes[0,0].legend(loc='upper left')



sns.distplot(horror.avg_rating, ax=axes[0,1], norm_hist=True,

             color='r', axlabel=False, label=f'Total of {horror_n_movies} Movies')

axes[0,1].set_title(f'Horror (μ = {horror_rating_mean}, σ = {horror_rating_std}, Skp = {horror_rating_skp})',

                    fontsize=20)

axes[0,1].legend(loc='upper left')



sns.distplot(scifi.avg_rating, ax=axes[1,0], norm_hist=True,

             color='g', axlabel=False, label=f'Total of {scifi_n_movies} Movies')

axes[1,0].set_title(f'SciFi (μ = {scifi_rating_mean}, σ = {scifi_rating_std}, Skp = {scifi_rating_skp})',

                    fontsize=20)

axes[1,0].legend(loc='upper left')



sns.distplot(thriller.avg_rating, ax=axes[1,1], norm_hist=True,

             color='k', axlabel=False, label=f'Total of {thriller_n_movies} Movies')

axes[1,1].set_title(f'Thriller (μ = {thriller_rating_mean}, σ = {thriller_rating_std}, Skp = {thriller_rating_skp})', fontsize=20)

axes[1,1].legend(loc='upper left')



#plt.savefig('Figures/Avg_Rating_Distplot.png')

plt.show()
#Visualizing Outliers



fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10), squeeze=True)

fig.suptitle('Average Rating Summary per Genre', fontsize=25)



sns.boxplot(animation.avg_rating, ax=axes[0,0], color='m', orient='v')

axes[0,0].set_title('Animation', fontsize=20)



sns.boxplot(horror.avg_rating, ax=axes[0,1], color='r', orient='v')

axes[0,1].set_title('Horror', fontsize=20)



sns.boxplot(scifi.avg_rating, ax=axes[1,0], color='g', orient='v')

axes[1,0].set_title('SciFi', fontsize=20)



sns.boxplot(thriller.avg_rating, ax=axes[1,1], color='k', orient='v')

axes[1,1].set_title('Thriller', fontsize=20)



plt.show()
def outlier_fences(df, col):

    """Calculates the fence values to determine Outliers

    

    df: is a dataframe

    col: is the column header

    

    returns: List with upper and lower fence values"""

    

    q1 = df[col].quantile(0.25)

    q3 = df[col].quantile(0.75)

    

    iqr = q3 - q1

    

    lower = q1 - 1.5*iqr

    upper = q3 + 1.5*iqr

    

    return [lower, upper]
anim_fences = outlier_fences(animation, 'avg_rating')
worst_rated_anim = animation[animation.avg_rating < anim_fences[0]]

worst_rated_anim.sort_values('avg_rating').head()
best_rated_anim = animation[animation.avg_rating > anim_fences[1]]

best_rated_anim.sort_values('avg_rating').sort_values('avg_rating', ascending=False)
horror_fences = outlier_fences(horror, 'avg_rating')
worst_rated_horror = horror[horror.avg_rating < horror_fences[0]]

worst_rated_horror.sort_values('avg_rating').sort_values('avg_rating').head()
best_rated_horror = horror[horror.avg_rating > horror_fences[1]]

best_rated_horror.sort_values('avg_rating').sort_values('avg_rating', ascending=False)
scifi_fences = outlier_fences(scifi, 'avg_rating')
worst_rated_scifi = scifi[scifi.avg_rating < scifi_fences[0]]

worst_rated_scifi.sort_values('avg_rating').sort_values('avg_rating').head()
best_rated_scifi = scifi[scifi.avg_rating > scifi_fences[1]]

best_rated_scifi.sort_values('avg_rating').sort_values('avg_rating', ascending=False)
thriller_fences = outlier_fences(thriller, 'avg_rating')
worst_rated_thriller = thriller[thriller.avg_rating < thriller_fences[0]]

worst_rated_thriller.sort_values('avg_rating').sort_values('avg_rating').head()
best_rated_thriller = thriller[thriller.avg_rating > thriller_fences[1]]

best_rated_thriller.sort_values('avg_rating').sort_values('avg_rating', ascending=False)
#Grouping all movies by their release year and calculating the mean value of each feature by year

animation_by_year = animation.groupby('year').mean()[['avg_rating', 'likes', 'views']]

horror_by_year = horror.groupby('year').mean()[['avg_rating', 'likes', 'views']]

scifi_by_year = scifi.groupby('year').mean()[['avg_rating', 'likes', 'views']]

thriller_by_year = thriller.groupby('year').mean()[['avg_rating', 'likes', 'views']]
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10), squeeze=True)

fig.suptitle('Average Rating per Year', fontsize=25)



##Plots:



sns.scatterplot(data = animation_by_year.avg_rating, ax=axes[0,0], color='m',

               alpha=0.75)

axes[0,0].set_title('Animation', fontsize=20)

axes[0,0].set_ylabel('Average Rating')

axes[0,0].set_ylim((2.4,4))



sns.scatterplot(data = horror_by_year.avg_rating, ax=axes[0,1], color='r',

               alpha=0.75)

axes[0,1].set_title('Horror', fontsize=20)

axes[0,1].set_ylim((2.4,4))



sns.scatterplot(data = scifi_by_year.avg_rating, ax=axes[1,0], color='g',

               alpha=0.75)

axes[1,0].set_title('SciFi', fontsize=20)

axes[1,0].set_ylabel('Average Rating')

axes[1,0].set_xlabel('Year')

axes[1,0].set_ylim((2.4,4))



sns.scatterplot(data = thriller_by_year.avg_rating, ax=axes[1,1], color='k',

               alpha=0.75)

axes[1,1].set_title('Thriller', fontsize=20)

axes[1,1].set_xlabel('Year')

axes[1,1].set_ylim((2.4,4))



# -----Annotations-----: Annotated movies are just to point out some classics or personal favorites 



# ------------Animation------------

axes[0,0].annotate('Spirited \n Away', 

                   xy=(2001, animation_by_year.loc[2001]['avg_rating']),

                   xytext=(2001, animation_by_year.loc[2001]['avg_rating'] + 0.25),

                   arrowprops=dict(arrowstyle='fancy', facecolor='purple'),

                   horizontalalignment='left',

                   verticalalignment='bottom')



axes[0,0].annotate('Toy Story', 

                   xy=(1995, animation_by_year.loc[1995]['avg_rating']),

                   xytext=(1998, animation_by_year.loc[1995]['avg_rating'] - 0.4),

                   arrowprops=dict(arrowstyle='fancy', facecolor='purple'),

                   horizontalalignment='right',

                   verticalalignment='bottom')



axes[0,0].annotate('Grave of \n the Fireflies', 

                   xy=(1988, animation_by_year.loc[1988]['avg_rating']),

                   xytext=(1988, animation_by_year.loc[1988]['avg_rating'] + 0.3),

                   arrowprops=dict(arrowstyle='fancy', facecolor='purple'),

                   horizontalalignment='center',

                   verticalalignment='bottom')









# ---------- Horror ----------

axes[0,1].annotate('The Thing', 

                   xy=(1982, horror_by_year.loc[1982]['avg_rating']),

                   xytext=(1982, horror_by_year.loc[1982]['avg_rating'] + 0.35),

                   arrowprops=dict(arrowstyle='fancy', facecolor='red'),

                   horizontalalignment='center',

                   verticalalignment='top')



axes[0,1].annotate('Psycho', 

                   xy=(1960, horror_by_year.loc[1960]['avg_rating']),

                   xytext=(1960, horror_by_year.loc[1960]['avg_rating'] + 0.3),

                   arrowprops=dict(arrowstyle='fancy', facecolor='red'),

                   horizontalalignment='center',

                   verticalalignment='top')



axes[0,1].annotate('The \n VVitch', 

                   xy=(2015, horror_by_year.loc[2015]['avg_rating']),

                   xytext=(2015, horror_by_year.loc[2015]['avg_rating'] + 0.4),

                   arrowprops=dict(arrowstyle='fancy', facecolor='red'),

                   horizontalalignment='center',

                   verticalalignment='top')



# ---------- SciFi ----------



axes[1,0].annotate('Empire Strikes \n Back', 

                   xy=(1980, scifi_by_year.loc[1980]['avg_rating']),

                   xytext=(1980, scifi_by_year.loc[1980]['avg_rating'] - 0.25),

                   arrowprops=dict(arrowstyle='fancy', facecolor='green'),

                   horizontalalignment='center',

                   verticalalignment='top')



axes[1,0].annotate('2001: A \n Space Odyssey', 

                   xy=(1968, scifi_by_year.loc[1968]['avg_rating']),

                   xytext=(1968, scifi_by_year.loc[1968]['avg_rating'] + 0.5),

                   arrowprops=dict(arrowstyle='fancy', facecolor='green'),

                   horizontalalignment='center',

                   verticalalignment='top')



axes[1,0].annotate('Into the \n Spider-Verse', 

                   xy=(2018, scifi_by_year.loc[2018]['avg_rating']),

                   xytext=(2018, scifi_by_year.loc[2018]['avg_rating'] + 0.5),

                   arrowprops=dict(arrowstyle='fancy', facecolor='green'),

                   horizontalalignment='right',

                   verticalalignment='top')



# ---------Thriller-------------

axes[1,1].annotate('Parasite', 

                   xy=(2019, thriller_by_year.loc[2019]['avg_rating']),

                   xytext=(2019, thriller_by_year.loc[2019]['avg_rating'] + 0.3),

                   arrowprops=dict(arrowstyle='fancy', facecolor='black'),

                   horizontalalignment='right',

                   verticalalignment='top')



axes[1,1].annotate('Hitchcock Classic \n  Period', 

                   xy=(1955, thriller_by_year.loc[1955]['avg_rating'] ),

                   xytext=(1954, thriller_by_year.loc[1955]['avg_rating'] + 0.5),

                   arrowprops=dict(arrowstyle='-', facecolor='black'),

                   horizontalalignment='center',

                   verticalalignment='top')



axes[1,1].annotate('', 

                   xy=(1950, thriller_by_year.loc[1950]['avg_rating'] + 0.07),

                   xytext=(1950, thriller_by_year.loc[1950]['avg_rating'] + 0.34),

                   arrowprops=dict(arrowstyle='fancy', facecolor='black'),

                   horizontalalignment='left',

                   verticalalignment='top')



axes[1,1].annotate('', 

                   xy=(1960, thriller_by_year.loc[1960]['avg_rating'] + 0.05),

                   xytext=(1960, thriller_by_year.loc[1960]['avg_rating'] + 0.32),

                   arrowprops=dict(arrowstyle='fancy', facecolor='black'),

                   horizontalalignment='right',

                   verticalalignment='top')



axes[1,1].annotate('Oldboy', 

                   xy=(2003, thriller_by_year.loc[2003]['avg_rating'] ),

                   xytext=(2003, thriller_by_year.loc[2003]['avg_rating'] - 0.4),

                   arrowprops=dict(arrowstyle='fancy', facecolor='black'),

                   horizontalalignment='right',

                   verticalalignment='bottom')







#plt.savefig('Figures/Avg_Rating_per_Year.png')

plt.show()
top_animation = animation.nlargest(15, 'avg_rating')

top_animation
miyazaki_films = animation[animation['director'] == 'Hayao Miyazaki'].sort_values('avg_rating', ascending=False)

miyazaki_films['avg_rating'].mean()
animation.avg_rating.mean()
animation['avg_rating'].std()
horror.nlargest(15, 'avg_rating')
horror[horror['title'] == 'The Witch']
horror[horror['title'] == 'The Thing']
horror.avg_rating.mean()
horror.avg_rating.std()
scifi.nlargest(15, 'avg_rating')
scifi[scifi['title'] == 'The Empire Strikes Back']
scifi.avg_rating.mean()
scifi.avg_rating.std()
thriller.nlargest(15, 'avg_rating')
hitchcock_films = thriller[thriller['director'] == 'Alfred Hitchcock'].sort_values('avg_rating', ascending=False)
thriller[(thriller.year >= 1950) & (thriller.year <= 1960)].nlargest(15, 'avg_rating')
hitchcock_films[(hitchcock_films.year >= 1950) & (hitchcock_films.year <= 1960)]
thriller[(thriller.year >= 1950) & (thriller.year <= 1960)].avg_rating.mean()
hitchcock_films[(hitchcock_films.year >= 1950) & (hitchcock_films.year <= 1960)].avg_rating.mean()
thriller[(thriller.year >= 1910) & (thriller.year <= 1930)].shape
(4.6 - thriller_rating_mean)/thriller_rating_std
(3.97 - thriller[(thriller.year >= 1950) & (thriller.year <= 1960)].avg_rating.mean())/thriller[(thriller.year >= 1950) & (thriller.year <= 1960)].avg_rating.std()
# Obtaining total number of ratings for each movie and genre

animation['total_ratings'] = animation.half_star + animation.one_star + animation.one_half_star + animation.two_star + animation.two_half_star + animation.three_star + animation.three_half_star + animation.four_star + animation.four_half_star + animation.five_star

horror['total_ratings'] = horror.half_star + horror.one_star + horror.one_half_star + horror.two_star + horror.two_half_star + horror.three_star + horror.three_half_star + horror.four_star + horror.four_half_star + horror.five_star

scifi['total_ratings'] = scifi.half_star + scifi.one_star + scifi.one_half_star + scifi.two_star + scifi.two_half_star + scifi.three_star + scifi.three_half_star + scifi.four_star + scifi.four_half_star + scifi.five_star

thriller['total_ratings'] = thriller.half_star + thriller.one_star + thriller.one_half_star + thriller.two_star + thriller.two_half_star + thriller.three_star + thriller.three_half_star + thriller.four_star + thriller.four_half_star + thriller.five_star

# Like/views ratio to get a relative measure

animation['like_to_views_ratio'] = animation.likes/animation.views

horror['like_to_views_ratio'] = horror.likes/horror.views

scifi['like_to_views_ratio'] = scifi.likes/scifi.views

thriller['like_to_views_ratio'] = thriller.likes/thriller.views
print('Like/Views Statistics: \n',

      f'Anim: μ ={animation.like_to_views_ratio.mean()}, \sigma ={animation.like_to_views_ratio.std()} \n',

      f'Horror: μ = {horror.like_to_views_ratio.mean()}, \sigma = {horror.like_to_views_ratio.std()} \n',

      f'Scifi: μ = {scifi.like_to_views_ratio.mean()}, \sigma = {scifi.like_to_views_ratio.std()} \n',

      f'Thriller: μ = {thriller.like_to_views_ratio.mean()}, \sigma = {thriller.like_to_views_ratio.std()}')
animation['total_ratings_to_views_ratio'] = animation.total_ratings/animation.views

horror['total_ratings_to_views_ratio'] = horror.total_ratings/horror.views

scifi['total_ratings_to_views_ratio'] = scifi.total_ratings/scifi.views

thriller['total_ratings_to_views_ratio'] = thriller.total_ratings/thriller.views
print('Total_Ratings/Views Statistics: \n',

      f'Anim: μ ={animation.total_ratings_to_views_ratio.mean()}, \sigma ={animation.total_ratings_to_views_ratio.std()} \n',

      f'Horror: μ = {horror.total_ratings_to_views_ratio.mean()}, \sigma = {horror.total_ratings_to_views_ratio.std()} \n',

      f'Scifi: μ = {scifi.total_ratings_to_views_ratio.mean()}, \sigma = {scifi.total_ratings_to_views_ratio.std()} \n',

      f'Thriller: μ = {thriller.total_ratings_to_views_ratio.mean()}, \sigma = {thriller.total_ratings_to_views_ratio.std()}')
#calculating Skewness

animation_skp = round(animation.like_to_views_ratio.skew(), 2)

horror_skp = round(horror.like_to_views_ratio.skew(), 2)

scifi_skp = round(scifi.like_to_views_ratio.skew(), 2)

thriller_skp = round(thriller.like_to_views_ratio.skew(), 2)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10), squeeze=True)

fig.suptitle('Like/Views Distribution', fontsize=25)



sns.distplot(animation.like_to_views_ratio, ax=axes[0,0], norm_hist=True,

             color='y', axlabel=False, label=f'Total of {animation_n_movies} Movies')

axes[0,0].set_title(f'Animation (μ = {round(animation.like_to_views_ratio.mean(),2)}, σ = {round(animation.like_to_views_ratio.std(), 2)}, Skp = {animation_skp})',

                    fontsize=20)

axes[0,0].legend(loc='upper right')



sns.distplot(horror.like_to_views_ratio, ax=axes[0,1], norm_hist=True,

             color='r', axlabel=False, label=f'Total of {horror_n_movies} Movies')

axes[0,1].set_title(f'Horror (μ = {round(horror.like_to_views_ratio.mean(), 2)}, σ = {round(horror.like_to_views_ratio.std(), 2)}, Skp = {horror_skp})',

                    fontsize=20)

axes[0,1].legend(loc='upper right')



sns.distplot(scifi.like_to_views_ratio, ax=axes[1,0], norm_hist=True,

             color='g', axlabel=False, label=f'Total of {scifi_n_movies} Movies')

axes[1,0].set_title(f'SciFi (μ = {round(scifi.like_to_views_ratio.mean(), 2)}, σ = {round(scifi.like_to_views_ratio.std(), 2)}, Skp = {scifi_skp})',

                    fontsize=20)

axes[1,0].legend(loc='upper right')



sns.distplot(thriller.like_to_views_ratio, ax=axes[1,1], norm_hist=True,

             color='k', axlabel=False, label=f'Total of {thriller_n_movies} Movies')

axes[1,1].set_title(f'Thriller (μ = {round(thriller.like_to_views_ratio.mean(), 2)}, σ = {round(thriller.like_to_views_ratio.std(), 2)}, Skp = {thriller_skp})',

                    fontsize=20)

axes[1,1].legend(loc='upper right')



#plt.savefig('Figures/Like_Views_Ratio.png')

plt.show()
animation_skp2 = round(animation.total_ratings_to_views_ratio.skew(), 2)

horror_skp2 = round(horror.total_ratings_to_views_ratio.skew(), 2)

scifi_skp2 = round(scifi.total_ratings_to_views_ratio.skew(), 2)

thriller_skp2 = round(thriller.total_ratings_to_views_ratio.skew(), 2)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10), squeeze=True)

fig.suptitle('Ratings/Views Distribution', fontsize=25)



sns.distplot(animation.total_ratings_to_views_ratio, ax=axes[0,0], norm_hist=True,

             color='y', axlabel=False, label=f'Total of {animation_n_movies} Movies')

axes[0,0].set_title(f'Animation (μ = {round(animation.total_ratings_to_views_ratio.mean(),2)}, σ = {round(animation.total_ratings_to_views_ratio.std(), 2)}, Skp = {animation_skp2})',

                    fontsize=20)

axes[0,0].legend(loc='upper right')



sns.distplot(horror.total_ratings_to_views_ratio, ax=axes[0,1], norm_hist=True,

             color='r', axlabel=False, label=f'Total of {horror_n_movies} Movies')

axes[0,1].set_title(f'Horror (μ = {round(horror.total_ratings_to_views_ratio.mean(), 2)}, σ = {round(horror.total_ratings_to_views_ratio.std(), 2)}, Skp = {horror_skp2})',

                    fontsize=20)

axes[0,1].legend(loc='upper left')



sns.distplot(scifi.total_ratings_to_views_ratio, ax=axes[1,0], norm_hist=True,

             color='g', axlabel=False, label=f'Total of {scifi_n_movies} Movies')

axes[1,0].set_title(f'SciFi (μ = {round(scifi.total_ratings_to_views_ratio.mean(), 2)}, σ = {round(scifi.total_ratings_to_views_ratio.std(), 2)}, Skp = {scifi_skp2})',

                    fontsize=20)

axes[1,0].legend(loc='upper left')



sns.distplot(thriller.total_ratings_to_views_ratio, ax=axes[1,1], norm_hist=True,

             color='k', axlabel=False, label=f'Total of {thriller_n_movies} Movies')

axes[1,1].set_title(f'Thriller (μ = {round(thriller.total_ratings_to_views_ratio.mean(), 2)}, σ = {round(thriller.total_ratings_to_views_ratio.std(), 2)}, Skp = {thriller_skp2})',

                    fontsize=20)

axes[1,1].legend(loc='upper left')



#plt.savefig('Figures/total_ratings_Views_Ratio.png')

plt.show()
# Visualizing relationships between data.

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10), squeeze=True)

fig.suptitle('Likes/Views vs Average Rating', fontsize=25)



sns.scatterplot(x='like_to_views_ratio', y='avg_rating', data = animation, ax=axes[0,0], color='m',

               alpha=0.4)

axes[0,0].set_title('Animation', fontsize=20)

axes[0,0].set_ylabel('Average Rating')

axes[0,0].set_xlabel('')

axes[0,0].set_ylim(0,5)

axes[0,0].set_xlim(0,0.6)





sns.scatterplot(x='like_to_views_ratio', y='avg_rating', data = horror, ax=axes[0,1], color='r',

               alpha=0.4)

axes[0,1].set_title('Horror', fontsize=20)

axes[0,1].set_xlabel('')

axes[0,1].set_ylabel('')

axes[0,1].set_ylim(0,5)

axes[0,1].set_xlim(0,0.6)





sns.scatterplot(x='like_to_views_ratio', y='avg_rating', data = scifi, ax=axes[1,0], color='g',

               alpha=0.4)

axes[1,0].set_title('SciFi', fontsize=20)

axes[1,0].set_ylabel('Average Rating')

axes[1,0].set_xlabel('Like to Views Ratio')

axes[1,0].set_ylim(0,5)

axes[1,0].set_xlim(0,0.6)





sns.scatterplot(x='like_to_views_ratio', y='avg_rating', data = thriller, ax=axes[1,1], color='k',

               alpha=0.4)

axes[1,1].set_title('Thriller', fontsize=20)

axes[1,1].set_xlabel('Like to Views Ratio')

axes[1,1].set_ylabel('')

axes[1,1].set_ylim(0,5)

axes[1,1].set_xlim(0,0.6)



#plt.savefig('Figures/Corr_AvgRating_LikeViewRatio.png')
anim_grouped = animation.groupby('director')

n_movies = anim_grouped.size()

anim_directors = anim_grouped.mean()
anim_directors['n_movies'] = n_movies



anim_directors.describe()
anim_top15_dirs = anim_directors[anim_directors.n_movies > anim_directors.n_movies.quantile(0.75)].nlargest(15, ['avg_rating', 'n_movies', 'like_to_views_ratio'])
anim_top15_dirs = anim_top15_dirs[['avg_rating', 'n_movies', 'running_time', 'views', 'likes', 'like_to_views_ratio', 'total_ratings_to_views_ratio' ]].round(2)

anim_top15_dirs 
horror_grouped = horror.groupby('director')

horror_n_movies = horror_grouped.size()
horror_directors = horror_grouped.mean()
horror_directors['n_movies'] = horror_n_movies
horror_directors.describe()
horror_top15_dirs = horror_directors[horror_directors.n_movies > horror_directors.n_movies.quantile(0.75)].nlargest(15, ['avg_rating', 'n_movies', 'like_to_views_ratio'])
horror_top15_dirs = horror_top15_dirs[['avg_rating', 'n_movies', 'running_time', 'views', 'likes', 'like_to_views_ratio', 'total_ratings_to_views_ratio' ]].round(2)

horror_top15_dirs 
scifi_grouped = scifi.groupby('director')

scifi_n_movies = scifi_grouped.size()
scifi_directors = scifi_grouped.mean()
scifi_directors['n_movies'] = scifi_n_movies

scifi_directors.describe()
scifi_top15_dirs = scifi_directors[scifi_directors.n_movies > scifi_directors.n_movies.quantile(0.75)].nlargest(15, ['avg_rating', 'n_movies', 'like_to_views_ratio'])
scifi_top15_dirs = scifi_top15_dirs[['avg_rating', 'n_movies', 'running_time', 'views', 'likes', 'like_to_views_ratio', 'total_ratings_to_views_ratio' ]].round(2)

scifi_top15_dirs 
thriller_grouped = thriller.groupby('director')

thriller_n_movies = thriller_grouped.size()
thriller_directors = thriller_grouped.mean()
thriller_directors['n_movies'] = thriller_n_movies
thriller_directors.describe()
thriller_top15_dirs = thriller_directors[thriller_directors.n_movies > thriller_directors.n_movies.quantile(0.75)].nlargest(15, ['avg_rating', 'n_movies', 'like_to_views_ratio'])
thriller_top15_dirs = thriller_top15_dirs[['avg_rating', 'n_movies', 'running_time', 'views', 'likes', 'like_to_views_ratio', 'total_ratings_to_views_ratio' ]].round(2)

thriller_top15_dirs 
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.cluster import KMeans

from sklearn.mixture import GaussianMixture
#Setting a Pipeline for Standardizing and PCA



anim_pipeline = Pipeline([('scaler', StandardScaler()),

                         ('pca', PCA(n_components=0.94, random_state=69))], verbose=True)
anim_features = animation[['avg_rating', 'running_time', 'like_to_views_ratio', 'total_ratings_to_views_ratio']]

anim_pipeline.fit(anim_features)
anim_pipeline['pca'].explained_variance_ratio_ # 3 PCs were kept, enough to make cluster visualization possible preserving 94% of variance
anim_coefs = pd.DataFrame(anim_pipeline['pca'].components_, columns=anim_features.columns, index=['PC1', 'PC2', 'PC3'])

anim_coefs
anim_features_3d = anim_pipeline.transform(anim_features) #Transformed features ready for clustering in PC space.
anim_reduced = pd.DataFrame(anim_features_3d, columns=anim_coefs.index, index=anim_features.index)
anim_reduced
horror_pipeline = Pipeline([('scaler', StandardScaler()),

                            ('pca', PCA(n_components=0.93, random_state=69))], verbose=True)
horror_features = horror[['avg_rating', 'running_time', 'like_to_views_ratio', 'total_ratings_to_views_ratio']]
horror_pipeline.fit(horror_features)
horror_pipeline['pca'].explained_variance_ratio_ # 3 PCs were kept, enough to make cluster visualization possible preserving 93% of variance
horror_coefs = pd.DataFrame(horror_pipeline['pca'].components_, columns=horror_features.columns, index=['PC1', 'PC2', 'PC3'])

horror_coefs
horror_features_3d = horror_pipeline.transform(horror_features) #Transformed features ready for clustering in PC space.
horror_reduced = pd.DataFrame(horror_features_3d, columns=horror_coefs.index, index=horror_features.index)
horror_reduced
scifi_pipeline = Pipeline([('scaler', StandardScaler()),

                            ('pca', PCA(n_components=0.93, random_state=69))], verbose=True)



scifi_features = scifi[['avg_rating', 'running_time', 'like_to_views_ratio', 'total_ratings_to_views_ratio']]
scifi_pipeline.fit(scifi_features)
scifi_pipeline['pca'].explained_variance_ratio_ # 3 PCs were kept, enough to make cluster visualization possible preserving 93% of variance
scifi_coefs = pd.DataFrame(scifi_pipeline['pca'].components_, columns=scifi_features.columns, index=['PC1', 'PC2', 'PC3'])

scifi_coefs
scifi_features_3d = scifi_pipeline.transform(scifi_features) #Transformed features ready for clustering in PC space.

scifi_reduced = pd.DataFrame(scifi_features_3d, columns=scifi_coefs.index, index=scifi_features.index)
scifi_reduced
thriller_pipeline = Pipeline([('scaler', StandardScaler()),

                            ('pca', PCA(n_components=0.93, random_state=69))], verbose=True)



thriller_features = thriller[['avg_rating', 'running_time', 'like_to_views_ratio', 'total_ratings_to_views_ratio']]
thriller_pipeline.fit(thriller_features)
thriller_pipeline['pca'].explained_variance_ratio_ # 3 PCs were kept, enough to make cluster visualization possible preserving 93% of variance
thriller_coefs = pd.DataFrame(thriller_pipeline['pca'].components_, columns=thriller_features.columns, index=['PC1', 'PC2', 'PC3'])

thriller_coefs
thriller_features_3d = thriller_pipeline.transform(thriller_features) #Transformed features ready for clustering in PC space.

thriller_reduced = pd.DataFrame(thriller_features_3d, columns=thriller_coefs.index, index=thriller_features.index)
thriller_reduced
from sklearn.metrics import silhouette_score
anim_kmeans_per_k = [KMeans(n_clusters=k, random_state=69).fit(anim_reduced) for k in range(1,11)]



anim_inertias = [model.inertia_ for model in anim_kmeans_per_k]
plt.figure(figsize=(9,5))

plt.plot(range(1, 11), anim_inertias, "yo-")

plt.xlabel("$k$", fontsize=14)

plt.ylabel("Inertia", fontsize=14)

# plt.annotate('Elbow',

#              xy=(4, inertias[3]),

#              xytext=(0.55, 0.55),

#              textcoords='figure fraction',

#              fontsize=16,

#              arrowprops=dict(facecolor='black', shrink=0.1)

#             )

plt.xlim([0, 11])

#plt.savefig("Figures/inertia_vs_k_plot_anim.png")

plt.show()
anim_silhouette_scores = [silhouette_score(anim_reduced, model.labels_) for model in anim_kmeans_per_k[1:]]

anim_silhouette_scores
plt.figure(figsize=(9, 5))

plt.plot(range(2, 11), anim_silhouette_scores, "yo-")

plt.xlabel("$k$", fontsize=14)

plt.ylabel("Silhouette score", fontsize=14)

#plt.axis([1.8, 8.5, 0.55, 0.7])

#plt.savefig("Figures/silhouette_score_vs_k_plot.png")

plt.show()
from sklearn.metrics import silhouette_samples

from matplotlib.ticker import FixedLocator, FixedFormatter

import matplotlib as mpl
plt.figure(figsize=(15, 10))

n = 0

for k in (2, 3, 4, 7):

    n +=1

    plt.subplot(2, 2, n)

    

    y_pred = anim_kmeans_per_k[k - 1].labels_

    silhouette_coefficients = silhouette_samples(anim_reduced, y_pred)



    padding = len(anim_reduced) // 30

    pos = padding

    ticks = []

    for i in range(k):

        coeffs = silhouette_coefficients[y_pred == i]

        coeffs.sort()



        color = mpl.cm.Spectral(i / k)

        plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,

                          facecolor=color, edgecolor=color, alpha=0.7)

        ticks.append(pos + len(coeffs) // 2)

        pos += len(coeffs) + padding



    plt.gca().yaxis.set_major_locator(FixedLocator(ticks))

    plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))

    if k in (2, 4):

        plt.ylabel("Cluster")

        plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        

    if k in (4, 7):

        plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.xlabel("Silhouette Coefficient")

    else:

        plt.tick_params(labelbottom=False)



    plt.axvline(x=anim_silhouette_scores[k], color="red", linestyle="--")

    plt.title("$k={}$".format(k), fontsize=16)



#plt.savefig("Figures/silhouette_analysis_plot.png")

plt.show()
anim_clusters = anim_kmeans_per_k[2]
animation['cluster'] = anim_clusters.labels_
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure(figsize=(10,7))



ax = Axes3D(fig)

# Plotting Cluster 0

x_0 = anim_reduced[animation.cluster == 0].PC1

y_0 = anim_reduced[animation.cluster == 0].PC2

z_0 = anim_reduced[animation.cluster == 0].PC3

ax.scatter(x_0, y_0, z_0, label='Cluster 0', marker='o', alpha=0.5)



# Plotting Cluster 1

x_1 = anim_reduced[animation.cluster == 1].PC1

y_1 = anim_reduced[animation.cluster == 1].PC2

z_1 = anim_reduced[animation.cluster == 1].PC3

ax.scatter(x_1, y_1, z_1, label='Cluster 1', marker='o', alpha=0.5)



# Plotting Cluster 2

x_2 = anim_reduced[animation.cluster == 2].PC1

y_2 = anim_reduced[animation.cluster == 2].PC2

z_2 = anim_reduced[animation.cluster == 2].PC3

ax.scatter(x_2, y_2, z_2, label='Cluster 2', marker='o', alpha=0.5)









ax.set_xlabel('PC1')

ax.set_ylabel('PC2')

ax.set_zlabel('PC3')

ax.set_title('Animation: Clusters')



ax.view_init(15, 25)



plt.legend()



plt.show()

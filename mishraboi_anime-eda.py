import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

sns.set()

plt.rcParams['figure.dpi'] = 90

pd.set_option('display.max_rows', 500)
df_animes = pd.read_csv('../input/anime-data/Animes_eda.csv',index_col = 0)
df_animes = df_animes[['anime_id', 'anime_name', 'studio_id','studio_name', 'episodes_total',

       'source_material', 'air_date', 'overall_rating', 'members', 'synopsis',

       'number of tags', 'Action', 'Adventure', 'Cars', 'Comedy', 'Dementia',

       'Demons', 'Drama', 'Ecchi', 'Fantasy', 'Game', 'Harem', 'Hentai',

       'Historical', 'Horror', 'Josei', 'Kids', 'Magic', 'Martial_Arts',

       'Mecha', 'Military', 'Music', 'Mystery', 'Parody', 'Police',

       'Psychological', 'Romance', 'Samurai', 'School', 'Sci-Fi', 'Seinen',

       'Shoujo', 'Shoujo_Ai', 'Shounen', 'Shounen_Ai', 'Slice_of_Life',

       'Space', 'Sports', 'Super_Power', 'Supernatural', 'Thriller', 'Vampire',

       'Yaoi', 'Yuri' ]]

df_animes.head()
print(df_animes.dtypes)

df_animes.describe()
#number of animes

print('Animes',df_animes.shape[0])



#number of studios

print('Studios',len(pd.unique(df_animes['studio_id'])))



#tags

print('Genres',df_animes[['Action', 'Adventure', 'Cars', 'Comedy', 'Dementia',

       'Demons', 'Drama', 'Ecchi', 'Fantasy', 'Game', 'Harem', 'Hentai',

       'Historical', 'Horror', 'Josei', 'Kids', 'Magic', 'Martial_Arts',

       'Mecha', 'Military', 'Music', 'Mystery', 'Parody', 'Police',

       'Psychological', 'Romance', 'Samurai', 'School', 'Sci-Fi', 'Seinen',

       'Shoujo', 'Shoujo_Ai', 'Shounen', 'Shounen_Ai', 'Slice_of_Life',

       'Space', 'Sports', 'Super_Power', 'Supernatural', 'Thriller', 'Vampire',

       'Yaoi', 'Yuri' ]].shape[1])

df_animes['air_date'] = pd.to_datetime(df_animes['air_date'])

df_animes['air_date'].describe()
# Distribution of overall rating ~ Target variable

from statsmodels import robust

plt.figure(figsize=(7,5), dpi= 90)

sns.kdeplot(df_animes['overall_rating'],label = 'Overall Rating')

print('Mean: ',df_animes['overall_rating'].mean())

print('Median', np.median(df_animes['overall_rating']))

print('Standard Deviation: ',df_animes['overall_rating'].std())

print('MAD', robust.mad(df_animes['overall_rating']))
# High Rated animes (with rating greater than 75th Percentile)

perc_75 = np.round(np.percentile(df_animes['overall_rating'],75))

print('Number of high rated Animes (rating greater than 75th Percentile: ',df_animes[df_animes['overall_rating']>=7].shape[0])



# Top Animes of all time

print('These are the top 30 Animes of all time')

df_animes[['anime_name','overall_rating','air_date']].sort_values('overall_rating',ascending = False)[:30]
# Anime Production over time



import datetime

nat = np.datetime64('NaT')

def nat_check(nat):

    return nat == np.datetime64('NaT')    





yearly_ratings = df_animes[['anime_name','overall_rating','air_date']]

yearly_ratings['Year'] = yearly_ratings['air_date'].dt.year



# Number of Animes we have for every year

Ratings_time = yearly_ratings.groupby('Year')['overall_rating'].agg(['size','mean']).reset_index()



# Plotting

# Production over time

plt.figure(figsize = (10,5),dpi = 90)

plt.plot(Ratings_time['Year'],Ratings_time['size'],'r-')

plt.xticks(ticks=range(1917,2021,5),rotation = 60)

plt.xlabel('Year')

plt.ylabel('Count')

plt.title('Anime Production over time',fontdict={'fontweight':'bold'})

plt.show()

print('{} Animes were created in 2017.'.format(Ratings_time['size'].max()))



# Rating over time

# 5 year moving average ~ Year vs Overall Anime Rating



time_rating = yearly_ratings.groupby('Year').mean().reset_index()

time_rating['5MA'] = time_rating['overall_rating'].rolling(5).mean()



# Get the Peaks and Troughs

data = time_rating['5MA'].values

doublediff = np.diff(np.sign(np.diff(data)))

peak_locations = np.where(doublediff == -2)[0] + 1

doublediff2 = np.diff(np.sign(np.diff(-1*data)))

trough_locations = np.where(doublediff2 == -2)[0] + 1



# Plotting

plt.figure(figsize=(10,5), dpi = 90)

plt.plot(time_rating['Year'],time_rating['5MA'], label = '5 year MA')

plt.scatter(time_rating.Year[peak_locations],time_rating['5MA'][peak_locations], label = 'peaks',color = 'g')

plt.scatter(time_rating.Year[trough_locations],time_rating['5MA'][trough_locations], label = 'troughs',color = 'r')



# Annotation

for t, p in zip(trough_locations[::2], peak_locations[::4]):

    plt.text(time_rating.Year[p], time_rating['5MA'][p]+0.1, int(time_rating.Year[p]), horizontalalignment='center', color='darkgreen')

    plt.text(time_rating.Year[t], time_rating['5MA'][t]-0.3, int(time_rating.Year[t]), horizontalalignment='center', color='darkred')





# Decoration

plt.xticks(ticks=range(1917,2021,3), rotation = 60)

plt.title('Anime Rating Trend', fontdict={'fontweight':'bold'})

plt.xlabel('Year')

plt.ylabel('Rating')

plt.legend(loc = 'best')

plt.show()
# Printing out the best 5 animes from each band



for i in [1948, 1988 , 2010]:

    print('The 5 Year Moving Average for {} band was {}'.format(i,np.round(time_rating['5MA'][time_rating['Year'] == i].values[0],2)))



print('\n\n')    



print('Top 5 Animes of our best Years\n')

band1 = range(1944, 1948+1)

band2 =range(1984,1988+1)

band3 = range(2006,2010+1)



print('1944- 1948\n')

print(yearly_ratings[['anime_name',

                'overall_rating']][yearly_ratings['Year'].isin(band1)].sort_values('overall_rating',

                                                                                ascending = False)[:5],'\n')

print('1984- 1988\n')

print(yearly_ratings[['anime_name',

                'overall_rating']][yearly_ratings['Year'].isin(band2)].sort_values('overall_rating',

                                                                                ascending = False)[:5],'\n')



print('2006- 2010\n')

print(yearly_ratings[['anime_name',

                'overall_rating']][yearly_ratings['Year'].isin(band3)].sort_values('overall_rating',

                                                                                ascending = False)[:5],'\n')
#Number of animes per tag

tags = ['Action', 'Adventure', 'Cars', 'Comedy', 'Dementia',

       'Demons', 'Drama', 'Ecchi', 'Fantasy', 'Game', 'Harem', 'Hentai',

       'Historical', 'Horror', 'Josei', 'Kids', 'Magic', 'Martial_Arts',

       'Mecha', 'Military', 'Music', 'Mystery', 'Parody', 'Police',

       'Psychological', 'Romance', 'Samurai', 'School', 'Sci-Fi', 'Seinen',

       'Shoujo', 'Shoujo_Ai', 'Shounen', 'Shounen_Ai', 'Slice_of_Life',

       'Space', 'Sports', 'Super_Power', 'Supernatural', 'Thriller', 'Vampire',

       'Yaoi', 'Yuri' ]

#changing tags to category

df_animes[tags] = df_animes[tags].astype('category')



tags_num = {}

for i in tags:

    #print('Number of animes with {} tag {}'.format(i,sum(df_animes[i]==1)))

    tags_num[i] = sum(df_animes[i]==1)



tags_num_df = pd.DataFrame.from_dict(data=tags_num,orient = 'index',columns = ['Number of Animes'])



# plotting top and bottom ratings



#max

plt.subplot(211)

plt.bar(tags_num_df.sort_values('Number of Animes',ascending=False).head(10).index,

        tags_num_df.sort_values('Number of Animes',ascending=False).head(10)['Number of Animes'])

plt.xlabel('Tag')

plt.ylabel('Number of Animes')

for i, val in enumerate( tags_num_df.sort_values('Number of Animes',ascending=False).head(10)['Number of Animes'].values):

    plt.text(i, val, float(val), horizontalalignment='center', 

             verticalalignment='bottom', fontdict={'fontweight':300, 'size':10})

plt.xticks(rotation = 45)

plt.title('Top 10 Tags with most Animes created')

plt.show()



#least

plt.subplot(212)



plt.bar(tags_num_df.sort_values('Number of Animes',ascending=True).head(10).index,

        tags_num_df.sort_values('Number of Animes',ascending=True).head(10)['Number of Animes'])

plt.xlabel('Tag')

plt.ylabel('Number of Animes')

for i, val in enumerate( tags_num_df.sort_values('Number of Animes',ascending=True).head(10)['Number of Animes'].values):

    plt.text(i, val, float(val), horizontalalignment='center', 

             verticalalignment='bottom', fontdict={'fontweight':500, 'size':10,'color':'black'})

plt.xticks(rotation = 45)

plt.title('Top 10 Tags with least Animes created')

plt.show()
#tags with max ratings

tags_rating = {}



for i in tags:

    tags_rating[i] = np.round(np.median(df_animes['overall_rating'][df_animes[i]==1]),2)



tags_rating = pd.DataFrame.from_dict(data = tags_rating, orient = 'index', 

                                     columns = ['Median Rating']).sort_values('Median Rating')



# plotting top and bottom ratings

fig,ax = plt.subplots(nrows = 1, ncols= 3, figsize = (20,3),dpi = 144)

ax[0].bar(tags_rating.sort_values('Median Rating',ascending=False).head(10).index,

        tags_rating.sort_values('Median Rating',ascending=False).head(10)['Median Rating'])

ax[0].set_xlabel('Tag')

ax[0].set_ylabel('Median Rating')

ax[0].tick_params(axis = 'x',rotation = 90)

ax[0].set_title('Top 10 highest rated tags', fontdict = {'fontweight':'bold'})

ax[0].set_ylim(0,7.5,0.2)

for i, val in enumerate( tags_rating.sort_values('Median Rating',ascending=False).head(10)['Median Rating'].values):

    ax[0].text(i, val, float(val), horizontalalignment='center', 

             verticalalignment='bottom', fontdict={'fontweight':300, 'size':10})

fig.show()





# taking the top 10 highest rated tags to construct their boxplots



tags_box = {}



for i in tags_rating.sort_values('Median Rating',ascending=False).head(10).index:

    tags_box[i] = df_animes['overall_rating'][df_animes[i]==1]



pd.DataFrame(tags_box).boxplot()

ax[2].set_title('Ratings vs Tags Boxplot', fontdict = {'fontweight':'bold'})

ax[2].set_xlabel('Tag')

ax[2].set_ylabel('Overall Ratings')

ax[2].tick_params(axis = 'x',rotation = 90)

ax[2].set_ylim(0,10,0.30)

fig.show()



# MAD

tags_mad = {}

for i in tags_rating.sort_values('Median Rating',ascending=False).head(10).index:

    tags_mad[i] = np.round(stats.median_absolute_deviation(df_animes['overall_rating'][df_animes[i]==1]),2)



tags_mad = pd.DataFrame.from_dict(data = tags_mad, orient = 'index', 

                                     columns = ['MAD'])



ax[1].bar(tags_mad.index,

        tags_mad['MAD'],color = 'red')

for i, val in enumerate(tags_mad['MAD']):

    ax[1].text(i, val, float(val), horizontalalignment='center', 

             verticalalignment='bottom', fontdict={'fontweight':300, 'size':10})



ax[1].set_title('Rating deviation for top tags', fontdict = {'fontweight':'bold'})

ax[1].set_xlabel('Tag')

ax[1].tick_params(axis = 'x',rotation = 90)

ax[1].set_ylabel('Median Absolute Deviation')

fig.show()
#IQR

print('Psychological IQR', np.round(stats.iqr(df_animes['overall_rating'][df_animes['Psychological']==1], interpolation = 'midpoint'),2))

print('Thriller IQR', np.round(stats.iqr(df_animes['overall_rating'][df_animes['Thriller']==1], interpolation = 'midpoint'),2))

print('Police IQR', np.round(stats.iqr(df_animes['overall_rating'][df_animes['Police']==1], interpolation = 'midpoint'),2))

print('Harem IQR', np.round(stats.iqr(df_animes['overall_rating'][df_animes['Harem']==1], interpolation = 'midpoint'),2))
# Tags which occur most together



# Takking the top 10 tags only

tags_top10 = list(tags_box.keys())

tags_corr = df_animes[tags].astype(int).corr().sort_values(tags,ascending = False,axis=0)



# Plotting

plt.figure(figsize=(20,8), dpi= 144)

sns.heatmap(tags_corr[tags_top10], 

            xticklabels = tags_corr[tags_top10].columns, 

            yticklabels = tags_corr.index, cmap='RdYlGn', center=0, annot=True,linewidth=.3)



# Decorations

plt.title('Correlogram of Top 10 highest rated tags', fontsize=22)

plt.xticks(fontsize=12,rotation = 90)

plt.yticks(fontsize=12)

plt.show()
#Animes with scores of 0



zero_animes = df_animes[df_animes.overall_rating == 0]



# their tags

zero_rating_tags = {}



for i in tags:

    zero_rating_tags[i] = zero_animes[zero_animes[i]==1].shape[0]



zero_rating_tags = pd.DataFrame.from_dict(zero_rating_tags,orient = 'index',columns=['Number of animes'])
zero_rating_tags.sort_values('Number of animes',ascending=False).plot(kind = 'bar')

plt.show()
# studios which produce these kids animes

df_animes[df_animes.Kids==1].studio_name.value_counts()[:30].plot(color = 'coral',kind = 'bar')

plt.show()

df_animes[df_animes.Kids==1].studio_name.value_counts()[:30]

# Studios which have created the Highest Rated Animes

high_rated_studios = pd.DataFrame(df_animes[df_animes['overall_rating']>=perc_75].groupby('studio_name')['overall_rating'].

             count().rename('# Animes').sort_values(ascending = False))[1:31] #234 Unknowns

high_rated_studios['%total high rated animes'] = np.round(high_rated_studios['# Animes']*100/len(df_animes[df_animes['overall_rating']>=perc_75]),2)





print('10 Studios are responsible for creating',high_rated_studios[:10]['%total high rated animes'].sum(),'% of the total high rated Animes')

high_rated_studios
# Top studios via rating & #count

from sklearn.preprocessing import MinMaxScaler



studio_ratings = pd.DataFrame(df_animes.groupby('studio_name')['overall_rating'].agg(['median',

                                                                                      'count']).sort_values('median',

                                                                                                            ascending = False))



scaler = MinMaxScaler(feature_range=(0,5))

studio_ratings['scaled_count'] = scaler.fit_transform(studio_ratings['count'].values.reshape(-1,1))



# MCI(Median Count Index) = scaled_Median*scaled count

studio_ratings['MCI'] = studio_ratings['median']*studio_ratings['scaled_count']*100
# Plotting

fig,ax = plt.subplots(nrows = 1, ncols= 2, figsize = (20,10),dpi = 144)



plt.figure(figsize = (20,8),dpi=144)

# top 30 studios based on rating

ax[0].barh(studio_ratings['median'].sort_values(ascending = False).index[:30],

         studio_ratings['median'].sort_values(ascending = False)[:30],color = 'g')

ax[0].set_ylabel('Studio Name')

ax[0].set_xlabel('Average Ratings')

for i, val in enumerate(studio_ratings['median'].sort_values(ascending = False)[:30].values):

    ax[0].text(val+0.1, i, np.round(float(val),2), horizontalalignment='left', 

             verticalalignment='center', fontdict={'fontweight':'bold', 'size':10})

ax[0].set_title('Studios with Animes with highest Average Rating',fontdict = {'fontweight':'bold'})



# Count of Max rated studios

ax[1].bar(studio_ratings[:30].index,studio_ratings[:30]['count'],color = '#ffcc22')

       

        



for i, val in enumerate(studio_ratings[:30]['count'].values):

    ax[1].text(i, val+.5, np.round(float(val),2), horizontalalignment='center', 

             verticalalignment='bottom', fontdict={'fontweight':'bold', 'size':10})



ax[1].tick_params(axis = 'x',rotation = 90)

ax[1].set_xlabel('Studio Name')

ax[1].set_ylabel('# Animes Created')

ax[1].set_title('Animes created by Highest rated Studios',fontdict = {'fontweight':'bold'})



plt.show()
# ratings of most studios with most animes



fig,ax = plt.subplots(nrows = 1, ncols= 1, figsize = (10,8))



ax.barh((studio_ratings.sort_values('count',ascending = False)[1:31]).sort_values('median',ascending=False).index,

           (studio_ratings.sort_values('count',ascending = False)[1:31]).sort_values('median',ascending=False)['median'])



for i, val in enumerate((studio_ratings.sort_values('count',

                                                    ascending = False)[1:31]).sort_values('median',

                                                                                                  ascending=False)['median'].values):

    ax.text(val + 0.1, i, np.round(float(val),2), horizontalalignment='left', 

             verticalalignment='center', fontdict={'fontweight':'bold', 'size':10})

ax.set_ylabel('Studio Name')

ax.set_xlabel('Average Ratings')

ax.set_title('Average Ratings of most Active Studios',fontdict = {'fontweight':'bold'})

plt.show()



# Standard deviation of ratings of studios with most animes created

plt.figure(figsize=(10,8))

studio_ratings_dev = pd.DataFrame(df_animes.groupby('studio_name')['overall_rating'].std().rename('Deviation'))

studio_ratings_dev = studio_ratings_dev[studio_ratings_dev.index.isin((studio_ratings.sort_values('count',

                                                                    ascending = False)[1:31]).sort_values('median',

                                                                                                          ascending=False).index)]

studio_ratings_dev = studio_ratings_dev.sort_values('Deviation',ascending=False)                                       

plt.hlines(y=studio_ratings_dev.index, xmin=0, xmax=studio_ratings_dev['Deviation'],color = 'g' ,alpha=0.4, linewidth=5)



for i, val in enumerate(studio_ratings_dev['Deviation'].values):

    plt.text(val + 0.1, i, np.round(float(val),2), horizontalalignment='left', 

             verticalalignment='center', fontdict={'fontweight':'bold', 'size':10})



plt.xlabel('Deviation')

plt.ylabel('Studio Name')

plt.title('Deviation in ratings for the most active Studios')

plt.show()



# Number of Animes created by studios (excluding 6806 Unknowns)



studios_num = pd.DataFrame(df_animes.groupby('studio_name')['studio_name'].count().rename('# Animes Created').

                           sort_values(ascending = False))[1:31]

#top 30 active studios plotting



plt.figure(figsize=(10,8))



plt.barh(list(studios_num.index[::-1]), 

        studios_num['# Animes Created'][::-1],color = 'coral')

for i, val in enumerate(studios_num['# Animes Created'][::-1].values):

    plt.text(val+1, i, float(val), horizontalalignment='left', 

             verticalalignment='center', fontdict={'fontweight':'bold', 'size':12})



plt.xlabel('Number of Animes created',fontdict={'fontweight':'bold'})

plt.ylabel('Studio Name',fontdict={'fontweight':'bold'})

plt.title('Top 30 most active Studios',fontdict={'fontweight':'bold'})

plt.show()
# Source Material

df_animes['source_material'] = pd.Categorical(df_animes['source_material'])

print('We have animes created from {} different sources'.format( df_animes['source_material'].describe()[1]))



# Pie Chart 

plt.figure(figsize=(10,5),dpi = 144)

labels = pd.DataFrame(df_animes['source_material'].value_counts()).index

values = pd.DataFrame(df_animes['source_material'].value_counts()).values

plt.pie(values, explode=[0.1]+list([0]*14), shadow=False,startangle=90)

plt.legend(labels,loc = 'best')

plt.tight_layout()

plt.axis('equal')

plt.show()



source_perc = np.round(df_animes['source_material'].value_counts()*100/df_animes['source_material'].value_counts().sum(),2) 

print('The Original source constitutes of {} % of the total animes created,\nFollowed by Manga which constitutes of {} % of Animes.'.format(source_perc.values[0],source_perc.values[1]))



print('Light Novel adaptations constitute of {} %'.format(source_perc[source_perc.index=='Light novel'].values[0]))
# Source wise rating - What sort of anime adaptations fetch the highest ratings

plt.figure(figsize = (6,5),dpi = 90)



sns.boxplot(data=df_animes,x='source_material',y = 'overall_rating')

plt.xticks(rotation = 90)

plt.xlabel('Source')

plt.ylabel('Rating')

plt.title('Sources vs Rating')

plt.show()



print('Median Original',np.median(df_animes['overall_rating'][df_animes['source_material'] == 'Original']),' MAD: ', 

      stats.median_absolute_deviation(df_animes['overall_rating'][df_animes['source_material'] == 'Original']))



print('Median Manga',np.median(df_animes['overall_rating'][df_animes['source_material'] == 'Manga']),' MAD: ', 

      stats.median_absolute_deviation(df_animes['overall_rating'][df_animes['source_material'] == 'Manga']))



print('Median Light Novel',np.median(df_animes['overall_rating'][df_animes['source_material'] == 'Light novel']),' MAD: ', 

      stats.median_absolute_deviation(df_animes['overall_rating'][df_animes['source_material'] == 'Light novel']))
# Numerical Features - Correlation



num_features = ['anime_name', 'episodes_total', 'members', 'number of tags', 'overall_rating' ]

num_anime = df_animes[num_features]

sns.pairplot(num_anime, diag_kind='kde')

plt.show()

num_anime.corr() # 35% correlation between number of tags and the rating
import statsmodels.api as sm



# Rating vs #Tags

model1 = sm.OLS(num_anime['overall_rating'], sm.add_constant(num_anime['number of tags']))

results1 = model1.fit()

print('Ratings ~ Number of Tags Model\n')

print(results1.summary(),'\n\n')
# Rating vs #Tags+members+Total Episodes

print('Ratings ~ #Tags+members+Total Episodes')

X = sm.add_constant(num_anime[['number of tags','members','episodes_total']])

model2 = sm.OLS(num_anime['overall_rating'],X,missing='drop')

results2 = model2.fit()

print(results2.summary(),'\n\n')
# Creating a feature logmembers and fitting linear model (first standardising members as well)

print('Rating ~ logmembers + #tags + episodes_total\n')

X['logmembers'] = np.log(1 + X['members'] - X['members'].mean()/X['members'].std()*100)

model = sm.OLS(num_anime['overall_rating'],X[['logmembers','number of tags','episodes_total']],missing='drop').fit()

print(model.summary())



#plotting fitted values

sm.graphics.plot_fit(model,'logmembers')

plt.legend(loc = 'upper left')

plt.show()
# Hyped Animes

# Exploring members further - Animes with a high number of members (outliers)



upper_bound = np.percentile(df_animes['members'], 75) + 1.5*stats.iqr(df_animes['members'])

                                                         

hyped_animes = df_animes[df_animes['members']>= upper_bound].sort_values('members',ascending = False)



hyped_animes.head(100)
# tags of hyped animes

tags_rating = {}



for i in tags:

    tags_rating[i] = [np.round(np.median(hyped_animes['overall_rating'][hyped_animes[i]==1]),2),

                      hyped_animes['overall_rating'][hyped_animes[i]==1].shape[0] ]



tags_rating = pd.DataFrame.from_dict(data = tags_rating, orient = 'index', 

                                     columns = ['Median Rating','#Animes']).sort_values('#Animes', ascending = False)



#Plotting tags of the most hyped animes

plt.figure(figsize=(8,5),dpi = 90)

tags_rating['#Animes'][20::-1].plot(kind = 'barh', color = 'hotpink')



for i, val in enumerate(tags_rating[20::-1]['#Animes']):

    plt.text(val+0.3, i, int(val), horizontalalignment='left', 

             verticalalignment='center', fontdict={'fontweight':300, 'size':9})



plt.xlabel('Tags')

plt.ylabel('Number of Animes')

plt.title('Most occuring tags in the Hyped Animes')

plt.show()



# Studios which have created the most hyped animes

pd.DataFrame(hyped_animes.groupby('studio_name')['members'].sum().rename('Total Members')

            ).sort_values('Total Members', ascending = False).head(30).plot(kind = 'bar')

plt.show()
# movies & series ratings (movies => episodes = 1)



movies = df_animes[df_animes['episodes_total']==1].sort_values('overall_rating',ascending = False)



print('The median rating for Anime Series is {}'.format(df_animes['overall_rating'][df_animes['episodes_total']>1].median()))

print('The median rating for Anime Movies is {}'.format(df_animes['overall_rating'][df_animes['episodes_total']==1].median()))



plt.title('Number of Movies and Series')

plt.bar( ['Movies','Series'],

        [df_animes[df_animes['episodes_total']==1].shape[0],df_animes[df_animes['episodes_total']>1].shape[0]])



for i, val in enumerate([df_animes[df_animes['episodes_total']==1].shape[0],df_animes[df_animes['episodes_total']>1].shape[0]]):

    plt.text(i, val-0.1, int(val), horizontalalignment='center', 

             verticalalignment='bottom', fontdict={'fontweight':500, 'size':13})



plt.show()
# tags and studios of movies 



# Studios which have created the most Animes movies

pd.DataFrame(hyped_animes.groupby('studio_name')['members'].sum().rename('Total Members')

            ).sort_values('Total Members', ascending = False).head(30).plot(kind = 'bar')





# tags occuring most in Anime movies

tags_rating = {}



for i in tags:

    tags_rating[i] = [np.round(np.median(movies['overall_rating'][movies[i]==1]),2),

                      movies['overall_rating'][movies[i]==1].shape[0] ]



tags_rating = pd.DataFrame.from_dict(data = tags_rating, orient = 'index', 

                                     columns = ['Median Rating','#Animes']).sort_values('#Animes', ascending = False)



#Plotting tags of Anime Movies

plt.figure(figsize=(8,5),dpi = 90)

tags_rating['#Animes'][20::-1].plot(kind = 'barh', color = 'g')



for i, val in enumerate(tags_rating[20::-1]['#Animes']):

    plt.text(val+0.3, i, int(val), horizontalalignment='left', 

             verticalalignment='center', fontdict={'fontweight':300, 'size':9})



plt.xlabel('Tags')

plt.ylabel('Number of Animes')

plt.title('Most occuring tags in the Anime Movies')

plt.show()
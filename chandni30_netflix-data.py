# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

from pandas import DataFrame

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('//kaggle/input/netflix-shows/netflix_titles_nov_2019.csv'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
netflix = pd.read_csv('../input/netflix-shows/netflix_titles.csv')



netflix = pd.DataFrame(netflix)
cols = netflix.columns #A look at what all data is available for 

lst_cols = cols.tolist



lst_cols()
# taking the subset

netflix_1 = netflix.loc[:,['show_id','country','type','title', 'rating', 'release_year', 'listed_in']]



#expanding the country and listen_in columns since they have multiple CSV 

netflix_split_1= netflix_1.set_index(['show_id','type','title','rating', 'release_year','listed_in'])#fixing the columns to not be expand

netflix_split_2= netflix_split_1.stack()

netflix_split_3= netflix_split_2.str.split(',', expand=True)

netflix_split_4= netflix_split_3.stack()

netflix_split_5= netflix_split_4.unstack(-2)

netflix_split_6= netflix_split_5.reset_index(-1, drop=True)

netflix_split_7= netflix_split_6.reset_index()



#removing leading and trailing spaces 

netflix_split_7['country'] = netflix_split_7['country'].str.strip() 

#netflix_split_7['listed_in'] = netflix_split_7['listed_in'].str.strip() 
#plotting

fig = plt.figure(figsize =(11, 5))

pd.value_counts(netflix_split_7['country']).head(10).plot(kind = "barh", color = 'orange')

plt.gca().invert_yaxis()

plt.xlabel("Frequency")

plt.ylabel("Country")

plt.title("Top Countries on Netflix : Content-Wise")

val = pd.value_counts(netflix_split_7['country']).head(10).values[0:15]

for index, value in enumerate(pd.value_counts(netflix_split_7['country']).head(10)):

    plt.text(value, index, str(value))

plt.show()
# taking the subset

netflix_type = netflix_split_7.loc[:,['country','type','show_id']]

netflix_type = netflix_type.dropna()



# seperating total content into type of content using pivot

netflix_type_pivot = netflix_type.pivot_table(index = ['country'], columns =['type'], values = ['show_id'], aggfunc = 'count', fill_value = 0)



#converting pivot table to dataframe

netflix_type_pivot.columns = netflix_type_pivot.columns.droplevel(0)

netflix_type_pivot = netflix_type_pivot.reset_index().rename_axis(None, axis=1).set_index('country')





#sorting



netflix_type_pivot['Total'] = netflix_type_pivot['Movie'] + netflix_type_pivot['TV Show']

netflix_top_type =  netflix_type_pivot.sort_values(by= "Total" , ascending=False).head(10)

#top in Movie

netflix_top_movie_country = netflix_type_pivot.drop(['TV Show', 'Total'], axis = 1).sort_values(by='Movie', ascending=False).head(10)

#top in TV Show

netflix_top_TV_country = netflix_type_pivot.drop(['Movie', 'Total'], axis = 1).sort_values(by='TV Show', ascending=False).head(10)



ax = netflix_top_type.drop(['Total'], axis = 1).plot(kind = 'barh', stacked = True,color=('c','orange'), figsize = (17,6), fontsize= 'large')

plt.gca().invert_yaxis()

for rect in ax.patches:

    # Find where everything is located

    height = rect.get_height()

    width = rect.get_width()

    x = rect.get_x()

    y = rect.get_y()

 # The width of the bar is the data value and can be used as the label

    label_text = f'{width}'

    # ax.text(x, y, text)

    label_x = x + width 

    label_y = y + height / 2

    ax.text(label_x, label_y, label_text, ha = 'right',va = 'center', fontsize=10)



ax.legend( loc='center right', borderaxespad=30,fontsize = 'large' ,title = "Content",title_fontsize = 'large')    

ax.set_ylabel("Country", fontsize=20)

ax.set_xlabel("Count", fontsize=20)

ax.set_title("Shows classification on Netflix of Top 10 Countries (content-wise)",fontsize=20)

plt.show()
fig = plt.figure()



# Divide the figure into a 2x2 grid, and give me the first section

ax1 = fig.add_subplot(221)



# Divide the figure into a 2x2 grid, and give me the second section

ax2 = fig.add_subplot(222)



fig.tight_layout(pad = 1)

fig.suptitle("Top Country on Netflix",fontsize=20)



netflix_top_movie_country.plot(kind = 'barh',color=('c'), figsize = (17,10), fontsize= 'medium', ax = ax1).invert_yaxis()

for rect in ax1.patches:

    # Find where everything is located

    height = rect.get_height()

    width = rect.get_width()

    x = rect.get_x()

    y = rect.get_y()

 # The width of the bar is the data value and can be used as the label

    label_text = f'{width}'

    # ax.text(x, y, text)

    label_x = x + width 

    label_y = y + height / 2

    ax1.text(label_x, label_y, label_text, ha = 'right',va = 'center', fontsize=8)

ax1.legend( loc='center right', borderaxespad=5,fontsize = 'medium' ,title = "Content",title_fontsize = 'medium')    

ax1.set_ylabel("Country ", fontsize= 12)

ax1.set_xlabel("Count", fontsize=10)



netflix_top_TV_country.plot(kind = 'barh',color=('y'), figsize = (17,10), fontsize= 'medium', ax = ax2).invert_yaxis()

for rect in ax2.patches:

    # Find where everything is located

    height = rect.get_height()

    width = rect.get_width()

    x = rect.get_x()

    y = rect.get_y()

 # The width of the bar is the data value and can be used as the label

    label_text = f'{width}'

    # ax.text(x, y, text)

    label_x = x + width 

    label_y = y + height / 2

    ax2.text(label_x, label_y, label_text, ha = 'right',va = 'center', fontsize=8)

ax2.legend( loc='center right', borderaxespad=5,fontsize = 'medium' ,title = "Content",title_fontsize = 'medium')   

ax2.set_ylabel(" ")

ax2.set_xlabel("Count", fontsize=10)





plt.show()
#expanding the country and listen_in columns since they have multiple CSV 

netflix_genre_1= netflix_1.set_index(['show_id','type'])#fixing the columns to not be expand

netflix_genre_2= netflix_genre_1.stack()

netflix_genre_3= netflix_genre_2.str.split(',', expand=True)

netflix_genre_4= netflix_genre_3.stack()

netflix_genre_5= netflix_genre_4.unstack(-2)

netflix_genre_6= netflix_genre_5.reset_index(-1, drop=True)

netflix_genre_7= netflix_genre_6.reset_index()



#removing leading and trailing spaces 

netflix_genre_7['listed_in'] = netflix_genre_7['listed_in'].str.strip() 





netflix_genre = netflix_genre_7.loc[:,["type", "listed_in", "show_id"]]

netflix_genre.dropna()



# seperating total content into type of content using pivot

netflix_genre_pivot = netflix_genre.pivot_table(index = ['listed_in'], columns =['type'], values = ['show_id'], aggfunc = 'count', fill_value = 0)



#converting pivot table to dataframe

netflix_genre_pivot.columns = netflix_genre_pivot.columns.droplevel(0)

#sorting

#top in Movie

netflix_top_movie_genre = netflix_genre_pivot.drop(['TV Show'], axis = 1).sort_values(by='Movie', ascending=False).head(10)

#top in TV Show

netflix_top_TV_genre = netflix_genre_pivot.drop(['Movie'], axis = 1).sort_values(by='TV Show', ascending=False).head(10)



fig = plt.figure()

ax1 = fig.add_subplot(221)

ax2 = fig.add_subplot(222)

fig.tight_layout(pad = 2)





fig.suptitle("Top Genre on Netflix",fontsize=20)



netflix_top_movie_genre.plot(kind = 'barh',color=('orange'), figsize = (17,10), fontsize= 'medium', ax = ax1).invert_yaxis()

for rect in ax1.patches:

    # Find where everything is located

    height = rect.get_height()

    width = rect.get_width()

    x = rect.get_x()

    y = rect.get_y()

 # The width of the bar is the data value and can be used as the label

    label_text = f'{width}'

    # ax.text(x, y, text)

    label_x = x + width 

    label_y = y + height / 2

    ax1.text(label_x, label_y, label_text, ha = 'right',va = 'center', fontsize=8)

ax1.legend( loc='best', borderaxespad=5,fontsize = 'medium' ,title = "Type",title_fontsize = 'medium')    

ax1.set_ylabel("Genre", fontsize= 12)

ax1.set_xlabel("Count", fontsize=10)



netflix_top_TV_genre.plot(kind = 'barh',color=('y'), figsize = (17,10), fontsize= 'medium', ax = ax2).invert_yaxis()

for rect in ax2.patches:

    # Find where everything is located

    height = rect.get_height()

    width = rect.get_width()

    x = rect.get_x()

    y = rect.get_y()

 # The width of the bar is the data value and can be used as the label

    label_text = f'{width}'

    # ax.text(x, y, text)

    label_x = x + width 

    label_y = y + height / 2

    ax2.text(label_x, label_y, label_text, ha = 'right',va = 'center', fontsize=8)

ax2.legend( loc='best', borderaxespad=5,fontsize = 'medium' ,title = "Type",title_fontsize = 'medium')   

ax2.set_ylabel("")

ax2.set_xlabel("Count", fontsize=10)

# taking the subset

netflix_country_genre = netflix.loc[:,['show_id','country','type','listed_in']]





#expanding the country and listen_in columns since they have multiple CSV 

def explode(data, indices_to_set,split_by):

    data= data.set_index([indices_to_set]).stack().str.split(split_by, expand=True).stack().unstack(-2).reset_index(-1, drop=True).reset_index()

    print (data)

    return ["the data has been exploded"]



netflix_cg_0 = explode(netflix_country_genre,['show_id','type','listed_in'],',')



#removing leading and trailing spaces 

netflix_cg_7['country'] = netflix_cg_7['country'].str.strip() 



# seperating total content into type of content using pivot

netflix_cg_pivot = netflix_cg_7.pivot_table(index = ['listed_in','country'], columns =['type'], values = ['show_id'], aggfunc = 'count', fill_value = 0)





#converting pivot table to dataframe

netflix_cg_pivot.columns = netflix_cg_pivot.columns.droplevel(0)

netflix_cg_pivot = netflix_cg_pivot.reset_index().rename_axis(None, axis=1).set_index('country')



#sorting

#top in Movie

netflix_top_movie_cg = netflix_cg_pivot.drop(['TV Show'], axis = 1).sort_values(by='Movie', ascending=False).head(50)

#top in TV Show

netflix_top_TV_cg = netflix_cg_pivot.drop(['Movie'], axis = 1).sort_values(by='TV Show', ascending=False).head(50)



pd.DataFrame(netflix_top_TV_cg)

netflix_movie_trend = netflix_split_7.loc[:,['release_year','title','show_id','type']].drop_duplicates()

netflix_movie_trend = netflix_movie_trend.loc[netflix_movie_trend['release_year'] > 1995]

netflix_trend  = netflix_movie_trend.pivot_table(index = ['release_year'], columns = ['type'], values = ['show_id'], aggfunc = 'count', fill_value = 0)



#converting pivot table to dataframe

netflix_trend.columns = netflix_trend.columns.droplevel(0)

netflix_trend = netflix_trend.reset_index().rename_axis(None, axis=1).set_index('release_year')



#figure

fig = plt.figure()

netflix_trend.plot(kind = 'line', figsize = (9,5), fontsize= 'medium', marker = 'o')



plt.legend( loc='upper left', borderaxespad=2,fontsize = 'medium' ,title = "Type",title_fontsize = 'medium')    

plt.ylabel("Count", fontsize=10)

plt.xlabel("Release Year", fontsize=10)

plt.title("Content Trend on Netflix",fontsize=15)

plt.show()
netflix_country_trend = netflix_split_7.loc[:,['release_year','country','show_id']].drop_duplicates()

netflix_country_trend = netflix_country_trend.loc[netflix_country_trend['release_year'] > 1995]

netflix_ct  = pd.value_counts(netflix_split_7['country']).head(5)



netflix_ct = netflix_ct.reset_index().rename_axis(None, axis=1).set_index('country')



netflix_ct = netflix_ct.reset_index(drop = True).rename(columns = {'index':'country'})



netflix_country_trend['key'] =  1



# join the two, keeping all of df1's indices

joined = pd.merge(netflix_country_trend, netflix_ct, on=['country'], how='inner')

netflix_top_ct = joined[pd.notnull(joined['key'])][netflix_country_trend.columns].drop(['key'], axis = 1)



netflix_top_ct  = netflix_top_ct.pivot_table(index = ['release_year'], columns = ['country'], values = ['show_id'], aggfunc = 'count', fill_value = 0)

#converting pivot table to dataframe

netflix_top_ct.columns = netflix_top_ct.columns.droplevel(0)

netflix_top_ct = netflix_top_ct.reset_index().rename_axis(None, axis=1).set_index('release_year')

#netflix_top_ct

#figure

fig = plt.figure()

netflix_top_ct.plot(kind = 'line', figsize = (14,4), fontsize= 'medium')

plt.legend( loc='upper left', borderaxespad=2,fontsize = 'medium' ,title = "Country",title_fontsize = 'medium')    

plt.ylabel("Count", fontsize=10)

plt.xlabel("Release Year", fontsize=10)

plt.title("Top Country Trend on Netflix",fontsize=15)

plt.show()
netflix_rating = netflix_split_7.loc[:,['show_id', 'rating', 'listed_in']]

netflix_rating_pivot  = netflix_rating.pivot_table(index = ['rating'], values = ['show_id'], aggfunc = 'count', fill_value = 0).sort_values('show_id',ascending = False)

# taking the subset

netflix_cast_duration = netflix.loc[:,['show_id','type','listed_in','duration']]



# taking the subset

netflix_test= netflix.loc[:,['show_id','country','type','listed_in']]





#expanding the country and listen_in columns since they have multiple CSV 

def explode(data, indices_to_set,split_by):

    data= data.set_index([indices_to_set]).stack().str.split(split_by, expand=True).stack().unstack(-2).reset_index(-1, drop=True).reset_index()

    print (data)

    return ["the data has been exploded"]



netflix_cg_0 = explode(netflix_test,['show_id','type','listed_in'],',')

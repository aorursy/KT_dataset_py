import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import warnings
from wordcloud import WordCloud
dataframe = pd.read_csv('../input/netflix-shows/netflix_titles.csv')

dataframe.head()
dataframe.drop(columns = 'show_id', inplace = True)

dataframe.head(0)
dataframe['duration'] = dataframe['duration'].apply(lambda x: x.split(' ')[0])

dataframe['duration'] = pd.to_numeric(dataframe['duration'])

dataframe.head()
dataframe.dropna(subset = ['rating', 'date_added'], inplace = True)
"""
# Create new dataframe

dataframe_content = dataframe.groupby('country').count().sort_values('type', ascending = False)

dataframe_content.reset_index(inplace = True)

# PLOT

sns.set_style('whitegrid')
f, ax = plt.subplots(figsize=(11,7))
x_content = dataframe_content[['country', 'type']].head(10)['type']
y_content = dataframe_content[['country', 'type']].head(10)['country']
sns.barplot(dataframe_content.index, dataframe_content.values, palette="RdBu")
plt.gca().invert_yaxis()
plt.title('Amount of Content', fontsize=16)
plt.show()

"""

# Create Dataframe
content_type = dataframe.groupby('type').count()
content_type.reset_index(inplace=True)
content_type = content_type[['type', 'title']]
content_type.columns = ['type', 'count']

# PLOT
fig2, ax2 = plt.subplots(figsize=(25, 6))
colors = ['steelblue', 'lightsalmon']
ax2.pie(x=content_type['count'], startangle=90, explode=(0, 0.03), colors=colors, autopct='%1.1f%%', textprops={'fontsize': 12})
ax2.legend(labels=content_type['type'], loc='upper left')

plt.show()
dataframe_country = dataframe[~dataframe['country'].isna()]

countries = ['United States', 'India', 'United Kingdom', 'Japan']

# Country Dataframes

def country_type(country):
    dataframe_country_type = dataframe_country[dataframe_country['country'] == country]
    dataframe_country_type = dataframe_country_type.groupby('type').count()
    dataframe_country_type.reset_index(inplace = True)
    dataframe_country_type = dataframe_country_type[['type', 'title']]
    dataframe_country_type.columns = ['type', 'count']
    return dataframe_country_type

usa_type = country_type('United States')
india_type = country_type('India')
uk_type = country_type('United Kingdom')
japan_type = country_type('Japan')
# PLOT

fig3, ax3 = plt.subplots(figsize=(11, 7))
color1 = 'steelblue'
color2 = 'lightsalmon'

ax3.bar(x='USA', height=usa_type.iloc[0][1], color=color1)
ax3.bar(x='USA', height=usa_type.iloc[1][1], bottom=usa_type.iloc[0][1], color=color2)
ax3.bar(x='India', height=india_type.iloc[0][1], color=color1)
ax3.bar(x='India', height=india_type.iloc[1][1], bottom=india_type.iloc[0][1], color=color2)
ax3.bar(x='UK', height=uk_type.iloc[0][1], color=color1)
ax3.bar(x='UK', height=uk_type.iloc[1][1], bottom=uk_type.iloc[0][1], color=color2)
ax3.bar(x='Japan', height=japan_type.iloc[0][1], color=color1)
ax3.bar(x='Japan', height=japan_type.iloc[1][1], bottom=japan_type.iloc[0][1], color=color2)

ax3.legend(labels=usa_type['type'], loc='upper right', prop={'size': 15})
ax3.set_title('Content Type by Country', fontsize=15)

plt.show()
dataframe['rating'].value_counts()
# Create Dataframe

rating_type = dataframe['rating'].value_counts().reset_index()

# Define x and y
x_rating_type = rating_type['index']
y_rating_type = rating_type['rating']

# PLOT
fig4, ax4 = plt.subplots(figsize=(11,7))
ax4.tick_params(axis = 'x', rotation = 45)
ax4.bar(x = x_rating_type, height = y_rating_type, color = 'steelblue')
ax4.set_title('Rating Types (worldwide)', fontsize = 17)


# Annotation of Values
## {:.0f} Format float with no decimal places

for a,b in zip(x_rating_type, y_rating_type): 
    plt.annotate('{:.0f}%'.format(round(int(b)/y_rating_type.sum()*100,0)),
                 xy=(a,b), xytext=(-10,4), textcoords='offset points')

"""
(round(int(b)/y_rating_type.sum()*100,0))
(int(b)/y_rating_type.sum()*100,0)
y_rating_type.sum()*100
"""

plt.show()

# Create dataframe

usa = dataframe[dataframe['country'] == 'United States'] 
uk = dataframe[dataframe['country'] == 'United Kingdom'] 
japan = dataframe[dataframe['country'] == 'Japan'] 
india = dataframe[dataframe['country'] == 'India']

rating_type_usa = usa['rating'].value_counts().reset_index()
rating_type_uk = uk['rating'].value_counts().reset_index()
rating_type_japan = japan['rating'].value_counts().reset_index()
rating_type_india = india['rating'].value_counts().reset_index()


# Align the dataframes

for x in rating_type['index']:
    if not rating_type_usa['index'].str.match(x).any():
        rating_type_usa = rating_type_usa.append({'index': x, 'rating': 0}, ignore_index = True)

for x in rating_type['index']:
    if not rating_type_uk['index'].str.match(x).any():
        rating_type_uk = rating_type_uk.append({'index': x, 'rating': 0}, ignore_index = True)
        
for x in rating_type['index']:
    if not rating_type_japan['index'].str.match(x).any():
        rating_type_japan = rating_type_japan.append({'index': x, 'rating': 0}, ignore_index = True)
        
for x in rating_type['index']:
    if not rating_type_india['index'].str.match(x).any():
        rating_type_india = rating_type_india.append({'index': x, 'rating': 0}, ignore_index = True)

# PLOT 

fig5, ax5 = plt.subplots(figsize=(12,7))
ax5.tick_params(axis='x', rotation=45)

# Define y
y_rating_type_usa = rating_type_usa['rating']/rating_type_usa['rating'].sum()
y_rating_type_india = rating_type_india['rating']/rating_type_india['rating'].sum()
y_rating_type_uk = rating_type_uk['rating']/rating_type_uk['rating'].sum()
y_rating_type_japan = rating_type_japan['rating']/rating_type_japan['rating'].sum()

## PLOT
ax5.plot(x_rating_type, y_rating_type_usa, 'o-', color='steelblue', label='USA')
ax5.plot(x_rating_type, y_rating_type_india, 'o-', color='lightsalmon', label='India')
ax5.plot(x_rating_type, y_rating_type_uk, 'o-', color='olivedrab', label='UK')
ax5.plot(x_rating_type, y_rating_type_japan, 'o-', color='indianred', label='Japan')

# Label
ax5.set_title('Rating Type Comparison', fontsize=15)
ax5.set_ylabel('Ratio', fontsize=15)
ax5.legend(loc='upper right', prop={'size': 15})

plt.show()
dataframe.head(1)
# Format 'date_added'

dataframe['date_added'] = dataframe['date_added'].str.replace(',', '')
dataframe['date_added'] = dataframe['date_added'].str.strip()

# Create dataframe

dataframe['year'] = dataframe['date_added'].str.split('/').str[2]
dataframe_without_2020 = dataframe[~(dataframe['year'] == '2020')]
dataframe_added = dataframe_without_2020.groupby('year').agg('count')
dataframe_added.reset_index(inplace = True)
dataframe_added = dataframe_added[['year', 'type']]



# PLOT

fig6, ax6 = plt.subplots(figsize=(11,7))
ax6.bar(dataframe_added['year'], dataframe_added['type'], color='steelblue')
ax6.set_title('Content Added Over the Years', fontsize = 15)

# Annotate values

for a,b in zip(dataframe_added['year'], dataframe_added['type']):
    plt.annotate(str(b), xy = (a,b), xytext = (-11,4), textcoords = 'offset points')


plt.show()
dataframe_without_2020.columns
dataframe_without_2020.head()
# Create dataframe

dataframe2 = dataframe_without_2020[dataframe_without_2020['type'] == 'Movie']
dataframe3 = dataframe_without_2020[dataframe_without_2020['type'] == 'TV Show']

x2 = dataframe2.groupby('year').agg('count')
x2.reset_index(inplace = True)

x3 = dataframe3.groupby('year').agg('count')
x3.reset_index(inplace = True)
x2.head(5)
x3.head()
"""
# PLOT

fig7, ax7 = plt.subplots(figsize = (11,7))
ax7.plot(x2['year'], x2['type'], 'o-', color = 'steelblue') # Movie
ax7.plot(x3['year'], x3['type'], 'o-', color = 'salmon') # TV Show

# Max values

y_max_movies = max(x2['type'])
y_max_tv = max(x3['type'])
x_max_movies = x2.iloc[x2['type'].idxmax]['year']
x_max_tv = x3.iloc[x3['type'].idxmax]['year']

# Annotate max values

plt.annotate(str(y_max_movies), xy = (x_max_movies, y_max_movies), xytext = (0,5), textcoords = 'offset points')
plt.annotate(str(y_max_tv), xy = (x_max_tv, y_max_tv), xytext = (0,5), textcoords = 'offset points')

# Label

plt.yticks(np.arange(0, y_max_movies, step = 200))
ax7.legend(labels = ['Movies', 'TV Shows'], loc = 'lower right', prop = {'size' : 13})
ax7.set_title('Movies & TV Shows Added Over the Years', fontsize = 15)

plt.show()
"""
dataframe.head()
# Sort by 'release_year', and extend sortingg to 'type', 'title', and 'release_year'
dataframe_oldest_movies = dataframe.sort_values('release_year')[['type', 'title', 'release_year']]

# Mark as 'Movie'
dataframe_oldest_movies[dataframe_oldest_movies['type'] == 'Movie']

# Columns 'title' and 'release_year'
dataframe_oldest_movies = dataframe_oldest_movies[['title', 'release_year']]

dataframe_oldest_movies.head()
# Sort by 'release_year', and extend sortingg to 'type', 'title', and 'release_year'
dataframe_oldest_shows = dataframe.sort_values('release_year')[['type', 'title', 'release_year']]

# Mark as 'TV Show'
dataframe_oldest_shows[dataframe_oldest_shows['type'] == 'TV Show']

# Columns 'title' and 'release_year'
dataframe_oldest_shows = dataframe_oldest_shows[['title', 'release_year']]

dataframe_oldest_shows.head(5)
dataframe.head(1)
dataframe.director
# Create dataframe

dataframe_director = dataframe[~dataframe['director'].isna()]

# Countries (all)

dataframe_director_all = dataframe_director.groupby('director').count().sort_values('type', ascending = False)
dataframe_director_all.reset_index(inplace = True)
dataframe_director_all = dataframe_director_all[['director', 'type']].head(10)
dataframe_director_all = dataframe_director_all.sort_values('type')

# Countries

def country_director(country):
    dataframe_country_director = dataframe_director[dataframe_director['country'] == country]
    dataframe_country_director = dataframe_country_director.groupby('director').count().sort_values('type', ascending = False)
    dataframe_country_director.reset_index(inplace = True)
    dataframe_country_director = dataframe_country_director[['director','type']].head(10)
    dataframe_country_director = dataframe_country_director.sort_values('type')

    return dataframe_country_director
    
dataframe_director_usa = country_director('United States')
dataframe_director_japan = country_director('Japan')
dataframe_director_uk = country_director('United Kingdom')
dataframe_director_india = country_director('India')
# PLOT

fig8, ax8 = plt.subplots(2, 3, figsize=(17,12))
ax8[0, 0].barh(dataframe_director_all['director'], dataframe_director_all['type'], color = 'steelblue')
ax8[0, 0].set_title('Top 10 Directors Worldwide', fontsize = 15)

ax8[0, 1].barh(dataframe_director_usa['director'], dataframe_director_usa['type'], color = 'steelblue')
ax8[0, 1].set_title('Top 10 Directors USA', fontsize = 15)

ax8[0, 2].barh(dataframe_director_india['director'], dataframe_director_india['type'], color = 'steelblue')
ax8[0, 2].set_title('Top 10 Directors India', fontsize = 15)

ax8[1, 0].barh(dataframe_director_uk['director'], dataframe_director_uk['type'], color = 'steelblue')
ax8[1, 0].set_title('Top 10 Directors UK', fontsize = 15)

ax8[1, 1].barh(dataframe_director_japan['director'], dataframe_director_japan['type'], color = 'steelblue')
ax8[1, 1].set_title('Top 10 Directors Japan', fontsize = 15)

ax8[1, 2].axis('off')

fig8.tight_layout(pad = 2)
dataframe.columns
print('Are there any missing values? : ', dataframe.cast.isnull().values.any())
dataframe.cast
# FORMAT

dataframe_cast = dataframe[~dataframe['cast'].isna()]
cast = ', '.join(str(v) for v in dataframe_cast['cast'])
cast = cast.split(', ')
cast_list = []

for x in cast:
    cast_list.append((x.strip(), cast.count(x)))
cast_list = sorted(cast_list, key = lambda x : x[1], reverse = True)
cast_list = list(dict.fromkeys(cast_list))

# Create Dataframe

dataframe_cast_all = pd.DataFrame(cast_list, columns = ('actor','count'))
dataframe_cast_all = dataframe_cast_all.head()
dataframe_cast_all.sort_values('count', inplace = True)

## Countries
"""
def country_cast(country):
    dataframe_country_cast = dataframe_cast[dataframe_cast['country'] == country]
    dataframe_country_cast = ', '.join(str(v) for v in dataframe_country_cast['cast'])
    dataframe_country_cast = dataframe_country_cast.split(', ')
    
    cast_list1 = []
    
    for x in dataframe_country_cast:
        cast_list1.append((x.strip(), dataframe_country_cast(x)))
    
    cast_list1 = sorted(cast_list1, key = lambda x : x[1], reverse = True)
    cast_list1 = list(dict.fromkeys(cast_list1))
    cast_list1 = pd.DataFrame(cast_list1, columns = ('actor','count'))
    cast_list1 = cast_list1.head(10)
    cast_list1.sort_values('count', inplace = True)
    
    return cast_list1
"""
def country_cast(country):
    dataframe_country_cast = dataframe_cast[dataframe_cast['country'] == country]
    dataframe_country_cast = ', '.join(str(v) for v in dataframe_country_cast['cast'])
    dataframe_country_cast = dataframe_country_cast.split(', ')
    cast_list1 = []
    for x in dataframe_country_cast:
        cast_list1.append((x.strip(), dataframe_country_cast.count(x)))
    cast_list1 = sorted(cast_list1, key=lambda x: x[1], reverse=True)
    cast_list1 = list(dict.fromkeys(cast_list1))
    cast_list1 = pd.DataFrame(cast_list1, columns=('actor', 'count'))
    cast_list1 = cast_list1.head(10)
    cast_list1.sort_values('count', inplace=True)
    return cast_list1

dataframe_cast_usa = country_cast('United States')
dataframe_cast_japan = country_cast('Japan')
dataframe_cast_uk = country_cast('United Kingdom')
dataframe_cast_india = country_cast('India')
# PLOT
fig15, ax15 = plt.subplots(2, 3, figsize=(17,12))
ax15[0, 0].barh(dataframe_cast_all['actor'], dataframe_cast_all['count'], color='steelblue')
ax15[0, 0].set_title('Top 10 Actors Worldwide', fontsize=15)

ax15[0, 1].barh(dataframe_cast_usa['actor'], dataframe_cast_usa['count'], color='steelblue')
ax15[0, 1].set_title('Top 10 Actors USA', fontsize=15)

ax15[0, 2].barh(dataframe_cast_india['actor'], dataframe_cast_india['count'], color='steelblue')
ax15[0, 2].set_title('Top 10 Actors India', fontsize=15)

ax15[1, 0].barh(dataframe_cast_uk['actor'], dataframe_cast_uk['count'], color='steelblue')
ax15[1, 0].set_title('Top 10 Actors UK', fontsize=15)

ax15[1, 1].barh(dataframe_cast_japan['actor'], dataframe_cast_japan['count'], color='steelblue')
ax15[1, 1].set_title('Top 10 Actors Japan', fontsize=15)

ax15[1, 2].axis('off')

fig15.tight_layout(pad=2)
dataframe.head(3)
# Create Dataframe

dataframe_movies = dataframe[dataframe['type'] == 'Movie']

dataframe_movies.head()
# Countries

data_country_duration_all = dataframe_movies.groupby('duration').count()
data_country_duration_all.reset_index(inplace = True)
data_country_duration_all = data_country_duration_all[['duration','type']]
data_country_duration_all.columns = ['duration', 'count']
data_country_duration_all.sort_values('duration', inplace = True)
data_country_duration_all['rel'] = data_country_duration_all['count'] / data_country_duration_all['count'].sum()
data_country_duration_all['durcount'] = data_country_duration_all['duration'] * data_country_duration_all['count']

average_all_movies = data_country_duration_all['durcount'].sum() / data_country_duration_all['count'].sum()


def country_duration(country):
    dataframe_country_duration = dataframe[(dataframe['country'] == country) & (dataframe['type'] == 'Movie')]
    dataframe_country_duration = dataframe_country_duration.groupby('duration').count()
    dataframe_country_duration.reset_index(inplace = True)
    dataframe_country_duration = dataframe_country_duration[['duration','type']]
    dataframe_country_duration.columns = ['duration','count']
    dataframe_country_duration.sort_values('duration', inplace = True)
    dataframe_country_duration['rel'] = dataframe_country_duration['count'] / dataframe_country_duration['count'].sum()
    dataframe_country_duration['durcount']=dataframe_country_duration['duration'] * dataframe_country_duration['count']
    return dataframe_country_duration


# PLOT

fig9, ax9 = plt.subplots(figsize = (20,3))
ax9.plot(data_country_duration_all['duration'], data_country_duration_all['rel'], color = 'steelblue')
plt.axvline(x = average_all_movies, color = 'lightsalmon', linestyle = '--')

## Labels

ax9.set_title('Length of Movies in All Countries', fontsize = 15)
ax9.set_ylabel('Relative Distribution', fontsize = 15)
ax9.set_xlabel('Minutes', fontsize = 15)
ax9.legend(labels = ['duration','average duration'], loc = 'upper right', prop = {'size':15})


for x in range(4):
    for y in range(1):
        dataframe_count = country_duration(countries[x])
        fig10, ax10 = plt.subplots(figsize = (20,3))
        ### Plot
        ax10.plot(dataframe_count['duration'], dataframe_count['rel'], color = 'steelblue')
        ax10.set_title('Movie Lengths in ' + countries[x], fontsize = 15)
        ax10.set_ylabel('Relative Distribution', fontsize=15)
        ax10.set_xlabel('Minutes', fontsize = 15)
        
        average_movies = dataframe_count['durcount'].sum() / dataframe_count['count'].sum()
        plt.axvline(x = average_movies, color='lightsalmon', linestyle='--')
        ax10.legend(labels = ['duration','average duration'],loc = 'upper right', prop = {'size': 15})
        plt.show()
data_country_duration_all
print('Average', average_all_movies)











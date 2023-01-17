# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objects as go

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/netflix-shows/netflix_titles.csv")

df.head(2)
df.info()
df['date_added'] = pd.to_datetime(df['date_added'])

df['year_added'] = df['date_added'].dt.year
df_tv = df[df['type'] == 'TV Show']

df_movie = df[df['type'] == 'Movie']
# Show unique directors in tv shows 

len(df_tv.director.unique())
# Show unique directors in movies

len(df_movie.director.unique())
#number of movies 

len(df_movie.title.unique())
#number of tv shows  

len(df_tv.title.unique())
# we will ignore the 2020 bar as we have note completed 2020 

fig = go.Figure()

fig.add_trace(go.Bar(

    x=df_tv['year_added'].value_counts().keys(),

    y=df_tv['year_added'].value_counts().values,

    text = df_tv['year_added'].value_counts().values

))

fig.update_traces(texttemplate='%{text}', textposition='outside',marker_color='red')

fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')

fig.show()
min_year = int(df_tv['year_added'].min())

max_year = int(df_tv['year_added'].max())

min_year, max_year
#top 10 countries with all the content 

df_groupby_country = df.groupby('country')['show_id'].count().sort_values(ascending = False).head(10)

countries_list = []

count_list = []

for index, value in df_groupby_country.items():

    countries_list.append(index)

    count_list.append(value)

    

cars = {

    'country': countries_list,

    'count': count_list

}



df_groupby_country = pd.DataFrame(cars, columns = ['country', 'count'])

df_groupby_country.columns
fig = px.bar( df_groupby_country, x = "country", y = "count",

    text = 'count', color = 'count'

)

fig.update_traces(texttemplate='%{text}', textposition='outside')

fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')

fig.show()
rating_data = df.groupby('rating')[['show_id']].count()

fig = px.pie(df, values=rating_data['show_id'],names =rating_data.index )

fig.update_traces(textposition='outside', textfont_size=14)



fig.show()
year_3_data = df[df['year_added'] > 2016]



year_3_data = year_3_data.drop(['title', 'director', 'cast', 'country', 'release_year', 'duration', 'rating', 'listed_in', 'description', 'date_added'], axis = 1)



# year_3_data

movie_count = len(year_3_data[year_3_data.type == 'Movie'])

tv_show_count = len(year_3_data[year_3_data.type == 'TV Show'])



year_3_data_movie = year_3_data[df['type'] == 'Movie']

year_3_data_tv = year_3_data[df['type'] == 'TV Show']



menu_sub= year_3_data[(year_3_data["type"] == 'Movie') & (year_3_data["year_added"] == 2018)] 



# Make data:

group_names = ['Movies', 'TV Shows']

group_size = [movie_count, tv_show_count]

subgroup_names = ["'17-movie", "'18-movie", "'19-movie", "'17-tv", "'18-tv", "'19-tv"]

subgroup_size = [

    len(year_3_data[(year_3_data["type"] == 'Movie') & (year_3_data["year_added"] == 2017)]), 

    len(year_3_data[(year_3_data["type"] == 'Movie') & (year_3_data["year_added"] == 2018)]), 

    len(year_3_data[(year_3_data["type"] == 'Movie') & (year_3_data["year_added"] == 2019)]), 

    

    len(year_3_data[(year_3_data["type"] == 'TV Show') & (year_3_data["year_added"] == 2017)]),

    len(year_3_data[(year_3_data["type"] == 'TV Show') & (year_3_data["year_added"] == 2018)]),

    len(year_3_data[(year_3_data["type"] == 'TV Show') & (year_3_data["year_added"] == 2019)]),

]

data = [# Portfolio (inner donut)

        go.Pie(values=subgroup_size,

               labels=["2017 movie", "2018 movie", "2019 movie", "2017 tv", "2018 tv", "2019 tv"],

              domain={'x':[0.2,0.8], 'y':[0.1,0.9]},

               hole=0.4,

               direction='clockwise',

               sort=True,

               marker={'colors':['#','green','light blue','blue']}),

        # Individual components (outer donut)

        go.Pie(values=group_size,

               labels=group_names,

               domain={'x':[0.1,0.9], 'y':[0,1]},

               hole=0.75,

               direction='clockwise',

               sort=True,

               marker={'colors':[' yellow',' red']},

               showlegend=False)]

fig = go.Figure(data=data, layout={'title':'Donut plot'})

fig.update_traces(textposition='inside', textfont_size=12)

fig.show()
year_list = list(range(2007,2020))

movies_per_year = []

tv_shows_per_year = []

both = []

for i in year_list:

    tv_count = len(df_tv[df_tv['year_added'] == i])

    movie_count = len(df_movie[df_movie['year_added'] == i])

    tv_shows_per_year.append(tv_count)

    movies_per_year.append(movie_count)

    both.append(tv_count + movie_count )
data1 = go.Scatter(x = year_list, y = tv_shows_per_year, mode = 'lines+markers', name = 'TV Shows Count')

data2 = go.Scatter(x = year_list, y = movies_per_year, mode = 'lines+markers', name = 'movie Count')

data3 = go.Scatter(x = year_list, y = both, mode = 'lines+markers', name = 'both content Count')

data = [data1,data2, data3]



layout = go.Layout(

    title = 'TV Shows and movies per year',

    legend = dict (x = 0.1, y = 0.8, orientation = 'v')

)



fig = go.Figure(data, layout = layout)



fig.show()
# Content by year

pclass = df_tv['year_added'].value_counts().to_frame().reset_index().rename(columns={'index':'year_added','year_added':'Count'})



print(pclass)



fig1 = go.Figure(data=[go.Scatter(

    x = pclass['year_added'], 

    y = pclass['Count'],

    mode = 'markers',

    marker = dict(

        color = pclass['Count'],

        size = pclass['Count'] * 0.2,

        showscale = False

    ))])



# Use theme [plotly_dark, ggplot2, plotly_dark, seaborn, plotly, plotly_white, presentation, xgridoff]

fig1.layout.template = 'seaborn'



fig1.update_layout(title = 'Content by Year', xaxis_title = "Class", yaxis_title = "Count", title_x = 0.5)

fig1.show()
def show_movies_vs_tvshows(country = 'India'):



    year_list        = []

    movies_country   = []

    tv_shows_country = []

    total_content_count = []

    

    for year in range(min_year, max_year):

        year = int(year)



        movies_count_country = len(df.loc[(df['year_added'] == year) & (df.country == country) & (df.type == 'TV Show')])

        tv_shows_count_country = len(df.loc[(df['year_added'] == year) & (df.country == country) & (df.type == 'Movie')])



        year_list.append(year)

        movies_country.append(movies_count_country)

        tv_shows_country.append(tv_shows_count_country)

        total_content_count.append(movies_count_country + tv_shows_count_country)



    data_movies = go.Scatter(x = year_list, y = movies_country, mode = 'lines+markers', name = 'Movies ('+str(country) + ')')

    data_tv_shows = go.Scatter(x = year_list, y = tv_shows_country, mode = 'lines+markers', name = 'TV Shows ('+str(country) + ')')

    data_all_content = go.Scatter(x = year_list, y = total_content_count, mode = 'lines+markers', name = 'Movies and Tv shows ('+str(country) + ')')

    data = [data_movies, data_tv_shows,data_all_content ]



    layout = go.Layout(

        title = 'Movies vs TV Shows - '+str(country),

        legend = dict (x = 0.1, y = 0.9, orientation = 'h')

    )



    fig = go.Figure(data, layout = layout)



    fig.show()
show_movies_vs_tvshows()
show_movies_vs_tvshows('United States')
group_country_movies = df.groupby('country')['show_id'].count().sort_values(ascending = False)



countries_list = []

count_list = []

for index, value in group_country_movies.items():

    countries_list.append(index)

    count_list.append(value)

    

cars = {

    'country': countries_list,

    'count': count_list

}



df4 = pd.DataFrame(cars, columns = ['country', 'count'])


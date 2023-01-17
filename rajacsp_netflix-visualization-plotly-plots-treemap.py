# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
FILEPATH = '/kaggle/input/netflix-shows/netflix_titles.csv'
df = pd.read_csv(FILEPATH)
df.head()
df.info()
df.isnull().sum()
df.isnull().sum().sum()
import missingno as miss

import matplotlib.pyplot as plt



%matplotlib inline
miss.matrix(df)

plt.show()
miss.heatmap(df)
miss.dendrogram(df)
miss.bar(df.sample(len(df)))
# Let's convert added_date to year

df['date_added'] = pd.to_datetime(df['date_added'])

df['month_added'] = df['date_added'].dt.month

df['year_added'] = df['date_added'].dt.year



df['year_added'] = df['year_added'].fillna(2008)

df['month_added'] = df['month_added'].fillna(0)



# convert float to int

df['year_added'] = df['year_added'].astype(int)

df['month_added'] = df['month_added'].astype(int)
df.info()
df.head()
# Get only TV Shows

df_tv = df[df['type'] == 'TV Show']
import seaborn as sns

import matplotlib.pyplot as plt



ax = sns.barplot(

    x = df_tv['year_added'].value_counts().keys(), 

    y = df_tv['year_added'].value_counts().values

)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)

plt.show()
group_country_movies = df.groupby('country')['show_id'].count().sort_values(ascending = False).head(10)



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





sns.set_style("darkgrid", {"axes.facecolor": ".9"})

# possible styles: whitegrid, dark, white



sns.set_context("notebook")





ax = sns.barplot(x = "country", y = "count", data = df4)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)



# You can collect more aesthetics from here:

# https://seaborn.pydata.org/tutorial/aesthetics.html
# Let's create a donut pie chart with 



def show_donut_plot(col):

    

    rating_data = df.groupby(col)[['show_id']].count()

    plt.figure(figsize = (12, 8))

    plt.pie(rating_data['show_id'], autopct = '%1.0f%%', startangle = 140, pctdistance = 1.1, shadow = True)



    # create a center circle for more aesthetics to make it better

    gap = plt.Circle((0, 0), 0.5, fc = 'white')

    fig = plt.gcf()

    fig.gca().add_artist(gap)

    

    plt.axis('equal')

    plt.legend(df[col])

    

    plt.title('Donut Plot by ' +str(col), loc='center')

    

    plt.show()
show_donut_plot('rating')
# Check tv shows added every year

min_year = int(df_tv['year_added'].min())

max_year = int(df_tv['year_added'].max())
min_year, max_year
# Show Movies, TV Shows with subgroups



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

subgroup_names = ["'17", "'18", "'19", "'17", "'18", "'19"]

subgroup_size = [

    len(year_3_data[(year_3_data["type"] == 'Movie') & (year_3_data["year_added"] == 2017)]), 

    len(year_3_data[(year_3_data["type"] == 'Movie') & (year_3_data["year_added"] == 2018)]), 

    len(year_3_data[(year_3_data["type"] == 'Movie') & (year_3_data["year_added"] == 2019)]), 

    

    len(year_3_data[(year_3_data["type"] == 'TV Show') & (year_3_data["year_added"] == 2017)]),

    len(year_3_data[(year_3_data["type"] == 'TV Show') & (year_3_data["year_added"] == 2018)]),

    len(year_3_data[(year_3_data["type"] == 'TV Show') & (year_3_data["year_added"] == 2019)]),

]



# Create colors

a, b = [plt.cm.Blues, plt.cm.Reds]



# First Ring (outside)

fig, ax = plt.subplots()

ax.axis('equal')

mypie, _ = ax.pie(group_size, radius=1.3, labels=group_names, colors=[a(0.6), b(0.6)] )

plt.setp( mypie, width=0.3, edgecolor='white')



# Second Ring (Inside)

mypie2, _ = ax.pie(subgroup_size, radius=1.3-0.3, labels=subgroup_names, 

                labeldistance=0.7, 

                colors=[a(0.1), a(0.2), a(0.3), b(0.2), b(0.3), b(0.4), b(0.5)]

            )

plt.setp( mypie2, width=0.4, edgecolor='white')

plt.margins(0,0)



plt.show()
year_list = []

tv_shows_per_year = []



for year in range(min_year, max_year):

    year = int(year)

    

    c_year = len(df_tv[df_tv['year_added'] == year])

    

    year_list.append(year)

    tv_shows_per_year.append(c_year)
# Do a simple graph with Plotly

import plotly.graph_objects as go



data1 = go.Scatter(x = year_list, y = tv_shows_per_year, mode = 'lines+markers', name = 'TV Shows Count')



data = [data1]



layout = go.Layout(

    title = 'TV Shows per year',

    legend = dict (x = 0.1, y = 0.8, orientation = 'h')

)



fig = go.Figure(data, layout = layout)



fig.show()
# Content by year

pclass = df_tv['year_added'].value_counts().to_frame().reset_index().rename(columns={'index':'year_added','year_added':'Count'})





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
# Let's check US vs Canada

year_list             = []

tv_shows_per_year_us  = []

tv_shows_per_year_can = []

for year in range(min_year, max_year):

    year = int(year)



    count_year_us = len(df_tv.loc[(df_tv['year_added'] == year) & (df_tv.country == 'United States')])

    count_year_can = len(df_tv.loc[(df_tv['year_added'] == year) & (df_tv.country == 'Canada')])

    

    year_list.append(year)

    tv_shows_per_year_us.append(count_year_us)

    tv_shows_per_year_can.append(count_year_can)
data1 = go.Scatter(x = year_list, y = tv_shows_per_year_us, mode = 'lines+markers', name = 'TV Shows - USA')

data2 = go.Scatter(x = year_list, y = tv_shows_per_year_can, mode = 'lines+markers', name = 'TV Shows - Canada')



data = [data1, data2]



layout = go.Layout(

    title = 'TV Shows per year - USA vs Canada',

    legend = dict (x = 0.1, y = 0.9, orientation = 'h')

)



fig = go.Figure(data, layout = layout)



fig.show()
# Find top 10 countries and TV Shows

top_10_countries_se = df_tv.country.value_counts().head(10)
top_10_countries = []

for i, v in top_10_countries_se.items():

    top_10_countries.append(i)

    

print(top_10_countries)
# Let's check top 10 countries' TV shows



year_list                  = []

tv_shows_per_year_country  = {}



for country in top_10_countries:

    tv_shows_per_year_country[country] = []



for year in range(min_year, max_year):

    year = int(year)

    

    current_country = {}

    

    for country in top_10_countries:

        current_country[country] = len(df_tv.loc[(df_tv['year_added'] == year) & (df_tv.country == country)])

    

    year_list.append(year)



    for country in top_10_countries:

        tv_shows_per_year_country[country].append(current_country[country])

data_dict = {}

data = []



for country in top_10_countries:

    

    data_dict[country] = go.Scatter(

        x = year_list, 

        y = tv_shows_per_year_country[country], 

        mode = 'lines+markers', 

        name = str(country)

    )

    data.append(data_dict[country])



layout = go.Layout(

    title = 'TV Shows per year - Various Countries',

    legend = dict (x = 0.1, y = 0.9, orientation = 'h')

)



fig = go.Figure(data, layout = layout)



fig.show()
# Get countries by region [https://restcountries.eu/#api-endpoints-name]

import requests



# This method will get countries by region

def get_countries_by_region(region):

    

    # Find asean countries

    resp = requests.get('https://restcountries.eu/rest/v2/regionalbloc/'+str(region))



    # if resp.status_code != 200:

    #     raise ApiError('GET /tasks/ {}'.format(resp.status_code))



    countries = []

    for item in resp.json():

        

        if(item['name'] == 'United States of America'):

            item['name'] = 'United States'

        

        countries.append(item['name'])



    # Check matched countries

    matched_countries = []

    for i, v in df_tv.country.value_counts().items():



        for country in countries:

            if(country == i):

                matched_countries.append(country)

    

    return matched_countries



def show_graph_by_region(region, region_title):



    matched_countries = get_countries_by_region(region)

    

    year_list                  = []

    tv_shows_per_year_country  = {}



    for country in matched_countries:

        tv_shows_per_year_country[country] = []



    for year in range(min_year, max_year):

        year = int(year)



        current_country = {}



        for country in matched_countries:

            current_country[country] = len(df_tv.loc[(df_tv['year_added'] == year) & (df_tv.country == country)])



        year_list.append(year)



        for country in matched_countries:

            tv_shows_per_year_country[country].append(current_country[country])



    data_dict = {}

    data = []



    for country in matched_countries:



        data_dict[country] = go.Scatter(

            x = year_list, 

            y = tv_shows_per_year_country[country], 

            mode = 'lines+markers', 

            name = str(country)

        )

        data.append(data_dict[country])



    layout = go.Layout(

        title = 'TV Shows - '+ region_title,

        legend = dict (x = 0.1, y = 0.9, orientation = 'h')

    )



    fig = go.Figure(data, layout = layout)



    fig.show()
# Show Association of Southeast Asian Nations

show_graph_by_region('asean', 'Association of Southeast Asian Nations (ASEAN)')
# Show African Union region (au)

show_graph_by_region('au', 'African Union region (AU)')
# Show Pacific Alliance region (pa)

show_graph_by_region('pa', 'Pacific Alliance (PA)')
# Show NAFTA (North American Free Trade Agreement)

show_graph_by_region('nafta', 'North American Free Trade Agreement (NAFTA)')
# Show SAARC (South Asian Association for Regional Cooperation)

show_graph_by_region('SAARC', 'South Asian Association for Regional Cooperation (SAARC)')
# Get only Movies

df_movie = df[df['type'] == 'Movie']
df_movie.head()
# This method will get countries by region

def get_countries_by_region_for_movies(region):

    

    # Find asean countries

    resp = requests.get('https://restcountries.eu/rest/v2/regionalbloc/'+str(region))



    # if resp.status_code != 200:

    #     raise ApiError('GET /tasks/ {}'.format(resp.status_code))



    countries = []

    for item in resp.json():

        

        if(item['name'] == 'United States of America'):

            item['name'] = 'United States'

        

        countries.append(item['name'])



    # Check matched countries

    matched_countries = []

    for i, v in df_movie.country.value_counts().items():



        for country in countries:

            if(country == i):

                matched_countries.append(country)

    

    return matched_countries



def show_movie_graph_by_region(region, region_title):



    matched_countries = get_countries_by_region_for_movies(region)

    

    year_list                  = []

    tv_shows_per_year_country  = {}



    for country in matched_countries:

        tv_shows_per_year_country[country] = []



    for year in range(min_year, max_year):

        year = int(year)



        current_country = {}



        for country in matched_countries:

            current_country[country] = len(df_movie.loc[(df_movie['year_added'] == year) & (df_movie.country == country)])



        year_list.append(year)



        for country in matched_countries:

            tv_shows_per_year_country[country].append(current_country[country])



    data_dict = {}

    data = []



    for country in matched_countries:



        data_dict[country] = go.Scatter(

            x = year_list, 

            y = tv_shows_per_year_country[country], 

            mode = 'lines+markers', 

            name = str(country)

        )

        data.append(data_dict[country])



    layout = go.Layout(

        title = 'Movies - '+ region_title,

        legend = dict (x = 0.1, y = 0.9, orientation = 'h')

    )



    fig = go.Figure(data, layout = layout)



    fig.show()
# Show Association of Southeast Asian Nations

show_movie_graph_by_region('asean', 'Association of Southeast Asian Nations (ASEAN)')
# Show Pacific Alliance region (pa)

show_movie_graph_by_region('pa', 'Pacific Alliance (PA)')
# Show NAFTA (North American Free Trade Agreement)

show_movie_graph_by_region('nafta', 'North American Free Trade Agreement (NAFTA)')
# Let's check TV Shows and Movies



def show_movies_vs_tvshows(country = 'United States'):



    year_list        = []

    movies_country   = []

    tv_shows_country = []

    

    for year in range(min_year, max_year):

        year = int(year)



        movies_count_country = len(df.loc[(df['year_added'] == year) & (df.country == country) & (df.type == 'TV Show')])

        tv_shows_count_country = len(df.loc[(df['year_added'] == year) & (df.country == country) & (df.type == 'Movie')])



        year_list.append(year)

        movies_country.append(movies_count_country)

        tv_shows_country.append(tv_shows_count_country)



    data_movies = go.Scatter(x = year_list, y = movies_country, mode = 'lines+markers', name = 'Movies ('+str(country) + ')')

    data_tv_shows = go.Scatter(x = year_list, y = tv_shows_country, mode = 'lines+markers', name = 'TV Shows ('+str(country) + ')')



    data = [data_movies, data_tv_shows]



    layout = go.Layout(

        title = 'Movies vs TV Shows - '+str(country),

        legend = dict (x = 0.1, y = 0.9, orientation = 'h')

    )



    fig = go.Figure(data, layout = layout)



    fig.show()
show_movies_vs_tvshows('United States')
show_movies_vs_tvshows('Canada')
show_movies_vs_tvshows('United Kingdom')
#!pip install squarify

# !pip show squarify
# show a treemap



import squarify



df_type_series = df.groupby('type')['show_id'].count()



type_sizes = []

type_labels = []

for i, v in df_type_series.items():

    type_sizes.append(v)

    type_labels.append(i)

    



fig, ax = plt.subplots(1, figsize = (12,12))

squarify.plot(sizes=type_sizes, 

              label=type_labels, 

              alpha=.8 )

plt.axis('off')

plt.show()
# Let's create more treemap by converting the code as function

# I have used only top 20 item to avoid confusion



def show_treemap(col):

    df_type_series = df.groupby(col)['show_id'].count().sort_values(ascending = False).head(20)



    type_sizes = []

    type_labels = []

    for i, v in df_type_series.items():

        type_sizes.append(v)

        

        type_labels.append(str(i) + ' ('+str(v)+')')





    fig, ax = plt.subplots(1, figsize = (12,12))

    squarify.plot(sizes=type_sizes, 

                  label=type_labels[:10],  # show labels for only first 10 items

                  alpha=.2 )

    plt.title('TreeMap by '+ str(col))

    plt.axis('off')

    plt.show()
show_treemap('country')
show_treemap('year_added')
show_treemap('rating')
show_treemap('month_added')
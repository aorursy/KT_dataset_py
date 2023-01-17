import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings 

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

from matplotlib import rcParams

import matplotlib.cm as cm

import seaborn as sns



import collections

from wordcloud import WordCloud, STOPWORDS



import os

print(os.listdir("../input"))
migrants = pd.read_csv('../input/MissingMigrants-Global-2019-03-29T18-36-07.csv')

migrants.drop(['Web ID', 'URL'], axis = 1, inplace=True)

migrants.head()
migrants[migrants['Region of Incident'] == 'Mediterranean'].groupby(['Cause of Death'])['Total Dead and Missing'].sum().sort_values(ascending=False)[:15]
migrants.describe(exclude='O')
migrants.describe(exclude='number')
migrants.info()
# Convert string month into numerical one

migrants['Reported Month(Number)'] = pd.to_datetime(migrants['Reported Month'], format='%b').apply(lambda x: x.month)



migrants[migrants['Reported Year'] == 2014]['Reported Month(Number)'].min(), migrants[migrants['Reported Year'] == 2019]['Reported Month(Number)'].max()
migrants.loc[:, 'Minimum Estimated Number of Missing'].sum(), migrants.loc[:, 'Number of Survivors'].sum(), 
print(migrants.loc[:, 'Number of Children'].sum())

migrants.loc[:, 'Number of Children'].plot.box()
print(migrants.loc[:, 'Number of Males'].sum())

migrants.loc[:, 'Number of Males'].plot.box()
print(migrants.loc[:, 'Number of Females'].sum())

migrants.loc[:, 'Number of Females'].plot.box()
migrants.loc[:, 'Number of Survivors'].plot.box()
na_sum = []

for col in migrants.columns:

    na_sum.append(migrants[col].isna().sum())



migrants_df = pd.DataFrame({'cols':migrants.columns,

                            'total_na' : na_sum})



migrants_df = migrants_df.sort_values(by='total_na', ascending=False).drop(list(migrants_df[migrants_df.total_na == 0].index), axis=0)

migrants_df.plot.bar(x = 'cols', y = 'total_na', rot=85, fontsize=18)

del migrants_df
import folium

from folium.plugins import HeatMapWithTime





migrants['Location Coordinates'].fillna('0, 0', inplace = True) # initialize missing value into 0, 0  location

migrants['lat'] = migrants['Location Coordinates'].apply(lambda x: float(str(x).split(', ')[0]))

migrants['lon'] = migrants['Location Coordinates'].apply(lambda x: float(str(x).split(', ')[1]))



basemap = folium.folium.Map(location = [migrants['lat'].median(), migrants['lon'].median()], zoom_start = 2)



indexes = ['{}/{}'.format(month, year) for year in migrants['Reported Year'].unique()[::-1] for month in range(1, 13)]



heat_data = [[[row['lat'], row['lon'], row['Total Dead and Missing']] for _, row in migrants[migrants['Reported Year'] == year][migrants['Reported Month(Number)'] == month].iterrows()]

             for year in migrants['Reported Year'].unique()[::-1] for month in range(1, 13)]



HeatMapWithTime(heat_data, auto_play = True, index = indexes, display_index = indexes).add_to(basemap)

basemap.save('Animated heatmap of migrants death or missing from 2014 to 2019 by month')

basemap

migrants['Region of Incident'].value_counts().plot.bar(rot=80, fontsize=18)
migrants['Migration Route'].value_counts().plot.bar(rot=80, fontsize=18)
migrants['UNSD Geographical Grouping'].value_counts().plot.bar(rot=80, fontsize=18)
all_cause_death          = ' '.join(migrants['Cause of Death'].str.lower())

all_location_description = ' '.join(migrants['Location Description'].str.lower().fillna(' '))

all_information_source   = ' '.join(migrants['Information Source'].str.lower().fillna(' '))
def words_frequency(corpus):

    stopwords = STOPWORDS

    

    wordcloud = WordCloud(stopwords=stopwords, background_color="white", max_words=150).generate(corpus) 

    rcParams['figure.figsize'] = 10, 20

    plt.imshow(wordcloud)

    plt.axis("off")

    plt.show()

    

    # Split corpus into each words

    filtered_words = [word for word in corpus.split() if word not in stopwords]

    

    # Make counter object that have each count of word

    counted_words = collections.Counter(filtered_words)

    

    # Store most common words

    words = []

    counts = []

    for letter, count in counted_words.most_common(10):

        words.append(letter)

        counts.append(count)



    rcParams['figure.figsize'] = 20, 10        # set figure size



    plt.title('Top words in the corpus vs their count')

    plt.xlabel('Count')

    plt.ylabel('Words')

    plt.barh(words, counts, color=cm.rainbow(np.linspace(0, 1, 10)))
words_frequency(all_cause_death)
words_frequency(all_location_description)
words_frequency(all_information_source)
migrants.loc[:, ['Total Dead and Missing', 'Number Dead', 'Number of Survivors']].plot.kde()

migrants.loc[:, ['Number of Males', 'Number of Females', 'Number of Children']].plot.kde()

migrants.loc[:, ['Total Dead and Missing']].plot.kde()
migrants['Region of Incident'].value_counts()[:10], migrants['Migration Route'].value_counts()[:10], migrants['UNSD Geographical Grouping'].value_counts()[:10]
def col_frequency_with_df(df, col):

    corpus = ' '.join(df[col].str.lower())

    return words_frequency(corpus)
col_frequency_with_df(migrants.loc[migrants['UNSD Geographical Grouping'] == 'Northern Africa',  :], 'Cause of Death')
col_frequency_with_df(migrants.loc[migrants['UNSD Geographical Grouping'] == 'Northern America', :], 'Cause of Death')
col_frequency_with_df(migrants.loc[migrants['UNSD Geographical Grouping'] == 'Uncategorized',  :], 'Cause of Death')
col_frequency_with_df(migrants.loc[migrants['Region of Incident'] == 'US-Mexico Border',    :], 'Cause of Death')
col_frequency_with_df(migrants.loc[migrants['Region of Incident'] == 'North Africa',    :], 'Cause of Death')
col_frequency_with_df(migrants.loc[migrants['Region of Incident'] == 'Mediterranean',    :], 'Cause of Death')
col_frequency_with_df(migrants.loc[migrants['Migration Route'] == 'Central America to US',    :], 'Cause of Death')
col_frequency_with_df(migrants.loc[migrants['Migration Route'] == 'Central Mediterranean',    :], 'Cause of Death')
col_frequency_with_df(migrants.loc[migrants['Migration Route'] == 'Western Mediterranean',    :], 'Cause of Death')
col_frequency_with_df(migrants.loc[migrants['Migration Route'] == 'Eastern Mediterranean',    :], 'Cause of Death')
migrants['Reported Year'].value_counts().sort_index().plot.bar()
migrants['Migration Route'].value_counts()
col_frequency_with_df(migrants.loc[migrants['Reported Year'] == 2014,    :], 'Cause of Death')
col_frequency_with_df(migrants.loc[migrants['Reported Year'] == 2015,    :], 'Cause of Death')
col_frequency_with_df(migrants.loc[migrants['Reported Year'] == 2016,    :], 'Cause of Death')
col_frequency_with_df(migrants.loc[migrants['Reported Year'] == 2017,    :], 'Cause of Death')
col_frequency_with_df(migrants.loc[migrants['Reported Year'] == 2018,    :], 'Cause of Death')
col_frequency_with_df(migrants[migrants['Reported Year'] == 2014][migrants['Region of Incident'] == 'US-Mexico Border'], 'Cause of Death')
col_frequency_with_df(migrants[migrants['Reported Year'] == 2015][migrants['Region of Incident'] == 'US-Mexico Border'], 'Cause of Death')
col_frequency_with_df(migrants[migrants['Reported Year'] == 2016][migrants['Region of Incident'] == 'US-Mexico Border'], 'Cause of Death')
col_frequency_with_df(migrants[migrants['Reported Year'] == 2017][migrants['Region of Incident'] == 'US-Mexico Border'], 'Cause of Death')
col_frequency_with_df(migrants[migrants['Reported Year'] == 2018][migrants['Region of Incident'] == 'US-Mexico Border'], 'Cause of Death')
def unique_year(df, location_col, location_value, value_counts_col):

    for year in list(df['Reported Year'].unique())[::-1]:

        print(year)

        counts = df[df['Reported Year'] == year][df[location_col] == location_value][value_counts_col].value_counts()

        print(counts[:5])

        print('Total : {}'.format(counts.sum()))

        print('-' * 30)

        

def n_deadmissing_by_year(df, sum_col, location_col, location_value):

    """Return sum of values of some column using grouped values"""

    

    for year in list(df['Reported Year'].unique())[::-1]:

        print(year)

        Sum = df[df['Reported Year'] == year][df[location_col] == location_value].groupby(['Cause of Death'])[sum_col].sum().sort_values(ascending=False) 

        print(Sum[:15])

        print('Total : {}'.format(Sum.sum()))

        print('-' * 30)
unique_year(migrants, 'Region of Incident', 'US-Mexico Border', 'Cause of Death')
# Number of deaths by reason of death from 2014 to 2019

print(migrants[migrants['Region of Incident'] == 'US-Mexico Border'].groupby(['Cause of Death'])['Total Dead and Missing'].sum().sort_values(ascending=False)[:15])

print('@' * 30)



n_deadmissing_by_year(migrants, 'Total Dead and Missing', 'Region of Incident', 'US-Mexico Border')
col_frequency_with_df(migrants[migrants['Reported Year'] == 2014][migrants['Region of Incident'] == 'North Africa'], 'Cause of Death')
col_frequency_with_df(migrants[migrants['Reported Year'] == 2015][migrants['Region of Incident'] == 'North Africa'], 'Cause of Death')
col_frequency_with_df(migrants[migrants['Reported Year'] == 2016][migrants['Region of Incident'] == 'North Africa'], 'Cause of Death')
col_frequency_with_df(migrants[migrants['Reported Year'] == 2017][migrants['Region of Incident'] == 'North Africa'], 'Cause of Death')
col_frequency_with_df(migrants[migrants['Reported Year'] == 2018][migrants['Region of Incident'] == 'North Africa'], 'Cause of Death')
col_frequency_with_df(migrants[migrants['Reported Year'] == 2019][migrants['Region of Incident'] == 'North Africa'], 'Cause of Death')
unique_year(migrants, 'Region of Incident', 'North Africa', 'Cause of Death')
# Number of deaths by reason of death from 2014 to 2019 

print(migrants[migrants['Region of Incident'] == 'North Africa'].groupby(['Cause of Death'])['Total Dead and Missing'].sum().sort_values(ascending=False)[:15])

print('@' * 30)



# Number of deaths by reason of death by year from 2014 to 2019

n_deadmissing_by_year(migrants, 'Total Dead and Missing', 'Region of Incident', 'North Africa')
col_frequency_with_df(migrants[migrants['Reported Year'] == 2014][migrants['Region of Incident'] == 'Mediterranean'], 'Cause of Death')
col_frequency_with_df(migrants[migrants['Reported Year'] == 2015][migrants['Region of Incident'] == 'Mediterranean'], 'Cause of Death')
col_frequency_with_df(migrants[migrants['Reported Year'] == 2016][migrants['Region of Incident'] == 'Mediterranean'], 'Cause of Death')
col_frequency_with_df(migrants[migrants['Reported Year'] == 2017][migrants['Region of Incident'] == 'Mediterranean'], 'Cause of Death')
col_frequency_with_df(migrants[migrants['Reported Year'] == 2018][migrants['Region of Incident'] == 'Mediterranean'], 'Cause of Death')
col_frequency_with_df(migrants[migrants['Reported Year'] == 2019][migrants['Region of Incident'] == 'Mediterranean'], 'Cause of Death')
unique_year(migrants, 'Region of Incident', 'Mediterranean', 'Cause of Death')
# Number of deaths by reason of death from 2014 to 2019 

print(migrants[migrants['Region of Incident'] == 'Mediterranean'].groupby(['Cause of Death'])['Total Dead and Missing'].sum().sort_values(ascending=False)[:15])

print('@' * 30)



# Number of deaths by reason of death by year from 2014 to 2019

n_deadmissing_by_year(migrants, 'Total Dead and Missing', 'Region of Incident', 'Mediterranean')
import matplotlib as mpl



font = {'family' : 'monospace',

        'weight' : 'bold',

        'size'   : 18}



lines = {'linewidth' : 2}



mpl.rc('font', **font)

mpl.rc('lines', **lines)



f, axes = plt.subplots(6, 1, figsize=(7, 5), sharex=True)



for year, ax in zip(migrants['Reported Year'].unique(), axes):

    sns.barplot(x=list(pd.DataFrame(migrants[migrants['Reported Year'] == year]['Migration Route'].value_counts())[:5].index),  y='Migration Route',

                palette="rocket", ax=ax, data = pd.DataFrame(migrants[migrants['Reported Year'] == year]['Migration Route'].value_counts())[:5])

    ax.axhline(0, color="k", clip_on=False)

    ax.set_ylabel(year)



plt.xticks(rotation=45)

plt.rcParams.update({'font.size': 22})

plt.show()



del font, lines, f, axes
def years_value_counts(df, col):    

    

    # Store whole value_counts series with their particular year into list

    stats_of_years = [pd.DataFrame(df[df['Reported Year'] == year][col].value_counts()) for year in df['Reported Year'].unique()]

    

    # concat dfs with their corresponding column

    stats_of_years         =  pd.concat(stats_of_years, axis=1)

    stats_of_years.columns = df['Reported Year'].unique()

    stats_of_years.fillna(0, inplace=True)

    return stats_of_years
route_year   = years_value_counts(migrants, 'Migration Route')

UNSD_year    = years_value_counts(migrants, 'UNSD Geographical Grouping')

region_year  = years_value_counts(migrants, 'Region of Incident')
import plotly.graph_objs as go

import plotly            as py

from plotly.offline      import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



fig = go.Figure(data=[go.Bar(name = str(col), x = route_year.transpose().index, y = route_year.transpose()[col]) 

                      for col in route_year.transpose().columns],

                

                layout = dict(

                    xaxis = dict(

                        title = dict(text = 'Number of incidents on specific migration route by year', font = dict(size=18))),

                    barmode = 'stack'))



iplot(fig)
fig = go.Figure(data=[go.Bar(name = str(col), x = region_year.transpose().index, y = region_year.transpose()[col]) 

                      for col in region_year.transpose().columns],

                

                layout = dict(

                    xaxis = dict(

                        title = dict(text = 'Number of incidents on specific region by year', font = dict(size=18))),

                    barmode = 'stack'))



iplot(fig)
fig = go.Figure(data=[go.Bar(name = str(col), x = UNSD_year.transpose().index, y = UNSD_year.transpose()[col]) 

                      for col in UNSD_year.transpose().columns],

                

                layout = dict(

                    xaxis = dict(

                        title = dict(text = 'Number of incidents on UNSD geo grouping location by year', font = dict(size=18))),

                    barmode = 'stack'))



iplot(fig)
fig = go.Figure(data=[go.Bar(name = str(year), 

                             x = migrants[migrants['Reported Year'] == year]['Reported Month(Number)'].value_counts().sort_index().index, 

                             y = migrants[migrants['Reported Year'] == year]['Reported Month(Number)'].value_counts().sort_index()) 

                      for year in migrants['Reported Year'].unique()[::-1]],

                

                layout = dict(

                    xaxis = dict(

                        title = dict(text = 'Number of migrant incidents by month on specific year', font = dict(size=18))),

                    barmode = 'stack'))



iplot(fig)
del route_year, UNSD_year, region_year
pd.to_datetime(migrants['Reported Month'], format='%b').apply(lambda x: x.month).value_counts().sort_index().plot.bar()
months = [m for m in range(1,13)]

years  = migrants['Reported Year'].unique()[::-1]



fig = go.Figure(data=[go.Bar(

    name = str(year), x = months,

    

    y = [migrants[migrants['Reported Year'] == year][migrants['Reported Month(Number)'] == month]['Total Dead and Missing'].sum() 

         for month in months]) 

                      for year in years],

                

                layout = dict(

                    xaxis = dict(

                        title = dict(text = 'Number of missing or dead migrants by month on specific year', font = dict(size=18))),

                    barmode = 'group'))



iplot(fig)
fig = go.Figure(data=[go.Bar(

    name = str(year), x = [m for m in range(1,13)],

    

    # List of number of migrants missing or death of months by specific year

    y = [

        migrants[migrants['Reported Year'] == year][migrants['Reported Month(Number)'] == month]['Total Dead and Missing'].sum() 

        for month in [m for m in range(1,13)]

    ] 

) for year in migrants['Reported Year'].unique()[::-1]],

                

                layout = dict(

                    xaxis = dict(

                        title = dict(text = 'Number of missing or dead migrants by month on specific year', font = dict(size=18))),

                    barmode = 'stack'))



iplot(fig)
def groupbar_by_month(df, region_name, barmode = 'stack'):

    months = [m for m in range(1,13)]

    years  = df['Reported Year'].unique()[::-1]



    fig = go.Figure(data=[go.Bar(

        name = str(year), x = months,



        y = [df[df['Region of Incident'] == region_name][df['Reported Year'] == year][df['Reported Month(Number)'] == month]['Total Dead and Missing'].sum() 

             for month in months]) 

                          for year in years],

            

                    layout = dict(

                        xaxis = dict(

                            title = dict(text = 'Number of missing or dead migrants by month on specific year in {}'.format(region_name), font = dict(size=18))),

                        barmode = barmode))

    

    return iplot(fig)
groupbar_by_month(migrants, 'US-Mexico Border', barmode = 'stack')

groupbar_by_month(migrants, 'US-Mexico Border', barmode = 'group')
groupbar_by_month(migrants, 'North Africa', barmode = 'stack')

groupbar_by_month(migrants, 'North Africa', barmode = 'group')
groupbar_by_month(migrants, 'Mediterranean', barmode = 'stack')

groupbar_by_month(migrants, 'Mediterranean', barmode = 'group')
fig = go.Figure(data=[go.Bar(

    name = col, x = migrants['Reported Year'].unique()[::-1],

    

    y = [

        migrants[migrants['Reported Year'] == year][col].sum() 

        for year in migrants['Reported Year'].unique()[::-1]

    ]

) for col in ['Number of Females', 'Number of Males', 'Number of Children']],

                

                layout = dict(

                    xaxis = dict(

                        title = dict(text = 'Number of missing or dead male, female and children migrants by year', font = dict(size=18))),

                    barmode = 'stack'))



iplot(fig)

del fig
def mfc_death_by_year(df, region, barmode = 'stack'):

    fig = go.Figure(data=[go.Bar(

        name = col, x = df['Reported Year'].unique()[::-1],



        y = [

            df[df['Region of Incident'] == region][df['Reported Year'] == year][col].sum() 

            for year in df['Reported Year'].unique()[::-1]

        ] 

    ) for col in ['Number of Females', 'Number of Males', 'Number of Children']],



                    layout = dict(

                        xaxis = dict(

                            title = dict(text = 'Number of missing or dead male, female and children migrants by year in {}'.format(region), font = dict(size=18))),

                        barmode = barmode))



    return iplot(fig)
for region in migrants['Region of Incident'].value_counts().index:

    mfc_death_by_year(migrants, region)
for region in migrants['Region of Incident'].value_counts().index:

    mfc_death_by_year(migrants, region, barmode = 'group')
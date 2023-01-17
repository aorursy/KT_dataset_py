import pandas as pd

import warnings

warnings.filterwarnings('ignore')

import numpy as np

import folium

from folium.plugins import FastMarkerCluster, Fullscreen, MiniMap, HeatMap, HeatMapWithTime

import geopandas as gpd

from branca.colormap import LinearColormap

import os

import json

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import re

from wordcloud import WordCloud, STOPWORDS
def style_function(feature):

    """

    Customize maps

    """

    return {

        'fillColor': '#ffaf00',

        'color': 'grey',

        'weight': 1.5,

        'dashArray': '5, 5'

    }



def highlight_function(feature):

    """

    Customize maps

    """

    return {

        'fillColor': '#ffaf00',

        'color': 'black',

        'weight': 2,

        'dashArray': '5, 5'

    }



def format_spines(ax, right_border=True):

    """

    This function sets up borders from an axis and personalize colors

    

    Input:

        Axis and a flag for deciding or not to plot the right border

    Returns:

        Plot configuration

    """    

    # Setting up colors

    ax.spines['bottom'].set_color('#CCCCCC')

    ax.spines['left'].set_color('#CCCCCC')

    ax.spines['top'].set_visible(False)

    if right_border:

        ax.spines['right'].set_color('#CCCCCC')

    else:

        ax.spines['right'].set_color('#FFFFFF')

    ax.patch.set_facecolor('#FFFFFF')

    

def count_plot(feature, df, colors='Blues_d', hue=False, ax=None, title=''):

    """

    This function plots data setting up frequency and percentage in a count plot;

    This also sets up borders and personalization.

    

    Input:

        The feature to be counted and the dataframe. Other args are optional.

    Returns:

        Count plot.

    """    

    # Preparing variables

    ncount = len(df)

    if hue != False:

        ax = sns.countplot(x=feature, data=df, palette=colors, hue=hue, ax=ax, 

                           order=df[feature].value_counts().index)

    else:

        ax = sns.countplot(x=feature, data=df, palette=colors, ax=ax,

                           order=df[feature].value_counts().index)



    # Make twin axis

    ax2=ax.twinx()



    # Switch so count axis is on right, frequency on left

    ax2.yaxis.tick_left()

    ax.yaxis.tick_right()



    # Also switch the labels over

    ax.yaxis.set_label_position('right')

    ax2.yaxis.set_label_position('left')

    ax2.set_ylabel('Frequency [%]')

    frame1 = plt.gca()

    frame1.axes.get_yaxis().set_ticks([])



    # Setting up borders

    format_spines(ax)

    format_spines(ax2)



    # Setting percentage

    for p in ax.patches:

        x=p.get_bbox().get_points()[:,0]

        y=p.get_bbox().get_points()[1,1]

        ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 

                ha='center', va='bottom') # set the alignment of the text

    

    # Final configuration

    if not hue:

        ax.set_title(df[feature].describe().name + ' Counting plot', size=13, pad=15)

    else:

        ax.set_title(df[feature].describe().name + ' Counting plot by ' + hue, size=13, pad=15)  

    if title != '':

        ax.set_title(title)       

    plt.tight_layout()

    

def country_analysis(country_name, data, palette, colors_plot2, color_lineplot):

    """

    This function creates a dashboard with informations of terrorism in a certain country.

    Input:

        The function receives the name of the country, the dataset and color configuration

    Output:

        It returns a 4 plot dashboard.

    """

    # Preparing

    country = data.query('country_txt == @country_name')

    if len(country) == 0:

        print('Country did not exists in dataset')

        return 

    country_cities = country.groupby(by='city', as_index=False).count().sort_values('eventid', 

                                                                                   ascending=False).iloc[:5, :2]

    suicide_size = country['suicide'].sum() / len(country)

    labels = ['Suicide', 'Not Suicide']

    colors = colors_plot2

    

    country_year = country.groupby(by='iyear', as_index=False).sum().loc[:, ['iyear', 'nkill']]

    country_weapon = country.groupby(by='weaptype1_txt', as_index=False).count().sort_values(by='eventid',

                                                                                             ascending=False).iloc[:, 

                                                                                                                   :2]

    # Dashboard

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

    

    # Plot 1 - Top 5 terrorism cities

    sns.barplot(x='eventid', y='city', data=country_cities, ci=None, palette=palette, ax=axs[0, 0])

    format_spines(axs[0, 0], right_border=False)

    axs[0, 0].set_title(f'Top 5 {country_name} Cities With Most Terrorism Occurences')

    """for p in axs[0, 0].patches:

        width = p.get_width()

        axs[0, 0].text(width-290, p.get_y() + p.get_height() / 2. + 0.10, '{}'.format(int(width)), 

                ha="center", color='white')"""

    axs[0, 0].set_ylabel('City')

    axs[0, 0].set_xlabel('Victims')

    

    # Plot 2 - Suicide Rate

    center_circle = plt.Circle((0,0), 0.75, color='white')

    axs[0, 1].pie((suicide_size, 1-suicide_size), labels=labels, colors=colors_plot2, autopct='%1.1f%%')

    axs[0, 1].add_artist(center_circle)

    format_spines(axs[0, 1], right_border=False)

    axs[0, 1].set_title(f'{country_name} Terrorism Suicide Rate')

    axs[0, 0].set_ylabel('Victims')

    

    # Plot 3 - Victims through the years

    sns.lineplot(x='iyear', y='nkill', data=country_year, ax=axs[1, 0], color=color_lineplot)

    format_spines(axs[1, 0], right_border=False)

    axs[1, 0].set_xlim([1970, 2017])

    axs[1, 0].set_title(f'{country_name} Number of Victims Over Time')

    axs[1, 0].set_ylabel('Victims')

    

    # Plot 4 - Terrorism Weapons

    sns.barplot(x='weaptype1_txt', y='eventid', data=country_weapon, ci=None, palette=palette, ax=axs[1, 1])

    axs[1, 1].set_xticklabels(axs[1, 1].get_xticklabels(), rotation=90)

    axs[1, 1].set_xlabel('')

    axs[1, 1].set_ylabel('Count')

    format_spines(axs[1, 1], right_border=False)

    axs[1, 1].set_title(f'{country_name} Weapons Used in Attacks')

    

    plt.suptitle(f'Terrorism Analysis in {country_name} between 1970 and 2017', size=16)    

    plt.tight_layout()

    plt.subplots_adjust(top=0.90)

    plt.show()
terr = pd.read_csv('../input/globalterrorismdb_0718dist.csv', encoding='ISO-8859-1')

attribs = ['eventid', 'iyear', 'imonth', 'iday', 'extended', 'country_txt', 'region_txt', 'city', 

                        'latitude', 'longitude', 'specificity', 'summary', 'success', 'suicide', 'attacktype1_txt', 

                        'targtype1_txt', 'copr1', 'target1', 'natlty1_txt', 'gname', 'motive', 'nperps', 

                        'weaptype1_txt', 'nkill', 'nkillter', 'nwound', 'nwoundte', 'ishostkid', 'nhostkid']

terr_data = terr.loc[:, attribs]

terr_data.head()
terr_data['country_txt'] = terr_data['country_txt'].apply(lambda x: x.replace('United States', 

                                                                              'United States of America'))

terr_data['weaptype1_txt'] = terr_data['weaptype1_txt'].apply(lambda x: x.split()[0] if 'Vehicle' in x.split() else x)
url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data'

world_geo = f'{url}/world-countries.json'

json_data = gpd.read_file(f'{url}/world-countries.json')
country_data = terr_data.groupby(by=['country_txt'], 

                                 as_index=False).count().sort_values(by='eventid', ascending=False).iloc[:, :2]

nkill_data = terr_data.groupby(by=['country_txt'], 

                                 as_index=False).sum().sort_values(by='eventid', 

                                                                   ascending=False).loc[:, ['country_txt', 'nkill']]

temp_global = json_data.merge(country_data, left_on='name', right_on='country_txt', how='left').fillna(0)

global_data = temp_global.merge(nkill_data, left_on='name', right_on='country_txt', how='left').fillna(0)



m = folium.Map(

    location=[0, 0], 

    zoom_start=1.50,

    tiles='openstreetmap'

)



folium.Choropleth(

    geo_data=json_data,

    name='Ataques Terroristas',

    data=country_data,

    columns=['country_txt', 'eventid'],

    key_on='feature.properties.name',

    fill_color='OrRd',

    fill_opacity=0.7,

    line_opacity=0.2,

    nan_fill_color='white',

    nan_fill_opacity=0.9,

    legend_name='Terrorism Recorded 1970 - 2017',

    popup_function='Teste'

).add_to(m)



Fullscreen(

    position='topright',

    title='Expand me',

    title_cancel='Exit me',

    force_separate_button=True

).add_to(m)



folium.GeoJson(

    global_data,

    style_function=style_function,

    highlight_function=highlight_function,

    tooltip=folium.GeoJsonTooltip(fields=['name', 'eventid', 'nkill'],

                                  aliases=['Country:', 'Incidents:', 'Victims'],

                                  labels=True,

                                  sticky=True)

).add_to(m)



m.save('terrorism_incidents.html')

m
heat_data = terr_data.groupby(by=['latitude', 'longitude'], 

                                 as_index=False).count().sort_values(by='eventid', ascending=False).iloc[:, :3]



m = folium.Map(

    location=[33.312805, 44.361488], 

    zoom_start=2.5, 

    tiles='Stamen Toner'

)



HeatMap(

    name='Mapa de Calor',

    data=heat_data,

    radius=10,

    max_zoom=13

).add_to(m)



Fullscreen(

    position='topright',

    title='Expand me',

    title_cancel='Exit me',

    force_separate_button=True

).add_to(m)



m.save('terrorism_density.html')

m
year_list = []

for year in terr_data['iyear'].sort_values().unique():

    data = terr_data.query('iyear == @year')

    data = data.groupby(by=['latitude', 'longitude'], 

                        as_index=False).count().sort_values(by='eventid', ascending=False).iloc[:, :3]

    year_list.append(data.values.tolist())



m = folium.Map(

    location=[0, 0], 

    zoom_start=2.0, 

    tiles='Stamen Toner'

)



HeatMapWithTime(

    name='Terrorism Heatmap',

    data=year_list,

    radius=9,

    index=list(terr_data['iyear'].sort_values().unique())

).add_to(m)



m
month_index = [

    'jan/2017',

    'feb/2017',

    'mar/2017',

    'apr/2017',

    'may/2017',

    'jun/2017',

    'jul/2017',

    'aug/2017',

    'sep/2017',

    'oct/2017',

    'nov/2017',

    'dec/2017'

]



month_list = []

for month in terr_data.query('iyear==2017')['imonth'].sort_values().unique():

    data = terr_data.query('imonth == @month')

    data = data.groupby(by=['latitude', 'longitude'], 

                        as_index=False).sum().sort_values(by='imonth', 

                                                          ascending=True).loc[:, ['latitude', 

                                                                                   'longitude', 

                                                                                   'nkill']]

    month_list.append(data.values.tolist())



m = folium.Map(

    location=[0, 0], 

    zoom_start=1.5, 

    tiles='Stamen Toner'

)



HeatMapWithTime(

    name='Mapa de Calor',

    data=month_list,

    radius=4,

    index=month_index

).add_to(m)



m
fig, ax = plt.subplots(figsize=(12, 6))

count_plot('region_txt', terr_data, ax=ax, colors='autumn')

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

ax.set_title('Distribution of Attacks per Region (1970-2017)', size=15)

plt.show()
country_victims = terr_data.groupby(by='country_txt', as_index=False).sum().sort_values(by='nkill', 

                                                                      ascending=False).loc[:, ['country_txt', 

                                                                                               'nkill']]

country_victims = country_victims.iloc[:10, :]



terr_data_2017 = terr_data.query('iyear == 2017')

country_victims_2017 = terr_data_2017.groupby(by='country_txt', as_index=False).sum().sort_values(by='nkill', 

                                                                      ascending=False).loc[:, ['country_txt', 

                                                                                               'nkill']]

country_victims_2017 = country_victims_2017.iloc[:10, :]

country_victims_2017['country_txt'][16] = 'Central African Rep.'

country_victims_2017['country_txt'][22] = 'Dem. Rep. Congo'



fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))



sns.barplot(x='nkill', y='country_txt', data=country_victims, ci=None,

                 palette='autumn', ax=axs[0])

sns.barplot(x='nkill', y='country_txt', data=country_victims_2017, ci=None,

                 palette='autumn', ax=axs[1])



format_spines(axs[0], right_border=False)

format_spines(axs[1], right_border=False)

axs[0].set_title('Top 10 - Total Victims by Country (1970-2017)')

axs[1].set_title('Top 10 - Total Victims by Country (2017)')

axs[0].set_ylabel('')

axs[1].set_ylabel('')



for p in axs[0].patches:

    width = p.get_width()

    axs[0].text(width-4000, p.get_y() + p.get_height() / 2. + 0.10, '{}'.format(int(width)), 

            ha="center", color='white')



for p in axs[1].patches:

    width = p.get_width()

    axs[1].text(width-300, p.get_y() + p.get_height() / 2. + 0.10, '{}'.format(int(width)), 

            ha="center", color='white')



plt.show()
country_analysis(country_name='Iraq', data=terr_data, palette='summer', 

                 colors_plot2=['crimson', 'green'], color_lineplot='crimson')
country_analysis(country_name='United States of America', data=terr_data, palette='plasma', 

                 colors_plot2=['crimson', 'navy'], color_lineplot='navy')
country_analysis(country_name='Nigeria', data=terr_data, palette='summer', 

                 colors_plot2=['crimson', 'green'], color_lineplot='green')
country_analysis(country_name='Colombia', data=terr_data, palette='hot', 

                 colors_plot2=['crimson', 'gold'], color_lineplot='crimson')
country_analysis(country_name='Egypt', data=terr_data, palette='copper', 

                 colors_plot2=['crimson', 'brown'], color_lineplot='brown')
terr_data['summary'][:10]
temp_corpus = terr_data['summary'].dropna()

corpus = temp_corpus.apply(lambda x: x.split(': ')[-1]).values

print(f'We have {len(corpus)} elements on the corpus\n\n')

print(f'Example 1: \n{corpus[1]}\n')

print(f'Example 2: \n{corpus[-1]}')
for c in corpus:

    urls = re.findall('(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', c)

    if len(urls) == 0:

        pass

    else:

        print(f'Description: {list(corpus).index(c)} - Links: {urls}')
# Example

corpus[6977]
# Replacing sites and hiperlinks

corpus_wo_hiperlinks = []

for c in corpus:

    c = re.sub(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', 'link', c)

    corpus_wo_hiperlinks.append(c)

corpus_wo_hiperlinks[6977]
# Example of description with number

corpus_wo_hiperlinks[399]
# Replacing numbers

corpus_wo_numbers = []

for c in corpus_wo_hiperlinks:

    c = re.sub('\d+(?:\.\d*(?:[eE]\d+))?', 'number', c)

    corpus_wo_numbers.append(c)

corpus_wo_numbers[399]
# Example with special characteres

corpus_wo_numbers[1113]
# Replacing special characteres with whitespace

corpus_text = []

for c in corpus_wo_numbers:

    c = re.sub(r'\W', ' ', c)

    corpus_text.append(c)

corpus_text[1113]
# Removing additional whitespaces

corpus_after_regex = []

for c in corpus_text:

    c = re.sub(r'\s+', ' ', c)

    corpus_after_regex.append(c)

    

corpus_after_regex[1113]
cleaned_corpus = pd.Series(corpus_after_regex).apply(lambda x: x.lower())

cleaned_corpus = list(cleaned_corpus.values)

cleaned_corpus[990]
# Genereating wordcloud

text = ' '.join(cleaned_corpus)

stopwords = set(STOPWORDS)



wordcloud = WordCloud(stopwords=stopwords, background_color="white", collocations=False).generate(text)



plt.figure(figsize=(15, 15))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
# Created by me, Alexander Ilin

import pandas as pd

import matplotlib.pyplot as plt

import folium

from IPython.display import HTML

%matplotlib inline
# Load data into the frame

data = pd.read_csv('../input/Indicators.csv')
# Filtering actual information - 'Research and' will match Research and development expenditure (% of GDP) feature

sci_fi_filter = data['IndicatorName'].str.contains('Research and')

sci_fi = data[sci_fi_filter]



# Group countries and take value associated with last year available for each country

sci_fi_latest = sci_fi[sci_fi['Year'] == sci_fi.groupby(['CountryCode'])['Year'].transform(max)]

# Checking

minimal = sci_fi_latest['Value'].min()

maximal = sci_fi_latest['Value'].max()

print('Worst science financing is {} in {}\nBest science financing is {} in {}'.format(minimal, sci_fi_latest[sci_fi_latest['Value'] == minimal].values[0][0], maximal, sci_fi_latest[sci_fi_latest['Value'] == maximal].values[0][0]))

sci_fi_latest.sort_values('Value').tail()

# Create map with color varying from yellow to blue as financing of Science go from min to max

graph_fin = folium.Map()

graph_fin.choropleth(geo_path='/home/arleg/PycharmProjects/Courses/Edx/DS/5/world-countries.json',

                data=sci_fi_latest, columns=['CountryCode', 'Value'], key_on='feature.id', 

                 legend_name=sci_fi.iloc[0]['IndicatorName'], fill_color='YlGnBu', fill_opacity=0.7, line_opacity=0.2)

graph_fin.save('ScienceFinancing_year.html')

HTML('<iframe src=ScienceFinancing_year.html width=700 height=450></iframe>')
# Filtering actual information - 'Researchers' will match Researchers in R&D (per million people) feature

sci_res_filter = data['IndicatorName'].str.contains('Researchers')

sci_res = data[sci_res_filter]



# Group countries and take value associated with the latest year for each country

sci_res_latest = sci_res[sci_res['Year'] == sci_res.groupby(['CountryCode'])['Year'].transform(max)]

# Checking

minimal = sci_res_latest['Value'].min()

maximal = sci_res_latest['Value'].max()

print('Least number of researchers per million is {} in {}\nThe largest number of researchers per million is {} in {}'.format(minimal, sci_res_latest[sci_res_latest['Value'] == minimal].values[0][0], maximal, sci_res_latest[sci_res_latest['Value'] == maximal].values[0][0]))

sci_res_latest.sort_values(by='Value')
# Create map with color varying from yellow to blue as number of researchers per million go from min to max

graph_nos = folium.Map()

graph_nos.choropleth(geo_path='/home/arleg/PycharmProjects/Courses/Edx/DS/5/world-countries.json',

                data=sci_res_latest, columns=['CountryCode', 'Value'], key_on='feature.id', 

                 legend_name=sci_res.iloc[0]['IndicatorName'], fill_color='YlGnBu', fill_opacity=0.7, line_opacity=0.2)

graph_nos.save('NumberOfScientists_year.html')

HTML('<iframe src=NumberOfScientists_year.html width=700 height=450></iframe>')
# Filtering actual information - 'Scientific' will match Scientific and technical journal articles feature

sci_art_filter = data['IndicatorName'].str.contains('Scientific')

sci_art = data[sci_art_filter]



# Group countries and take latest results for every country

sci_art_latest = sci_art[sci_art['Year'] == sci_art.groupby(['CountryCode'])['Year'].transform(max)]
# Checking

minimal = sci_art_latest['Value'].min()

maximal = sci_art_latest['Value'].max()

print('The smallest number of scientific articles is {} in {}\nThe largest number of scientific articles is {} in {}'.format(minimal, sci_art_latest[sci_art_latest['Value'] == minimal].values[0][0], maximal, sci_art_latest[sci_art_latest['Value'] == maximal].values[0][0]))

# sci_art_latest.head()

sci_art_latest.sort_values(by='Value')
# To keep map informative I cut row with data about big unions of countries (e.g. World, High income)

non_vis_on_map = sci_art_latest['Value'] < 250000

sci_art_latest = sci_art_latest[non_vis_on_map]

print('The smallest number of scientific articles is {} in {}\nThe largest number of scientific articles is {} in {}'.format(minimal, sci_art_latest[sci_art_latest['Value'] == minimal].values[0][0], sci_art_latest['Value'].max(), sci_art_latest[sci_art_latest['Value'] == sci_art_latest['Value'].max()].values[0][0]))

# Create map with color varying from yellow to blue as number of articles go from min to max

graph_art = folium.Map()

graph_art.choropleth(geo_path='/home/arleg/PycharmProjects/Courses/Edx/DS/5/world-countries.json',

                data=sci_art_latest, columns=['CountryCode', 'Value'], key_on='feature.id', 

                 legend_name=sci_art.iloc[0]['IndicatorName'], fill_color='YlGnBu', fill_opacity=0.7, line_opacity=0.2)

graph_art.save('NumberOfArticles_year.html')

HTML('<iframe src=NumberOfArticles_year.html width=700 height=450></iframe>')
# To make screenshots from html choropleths

import time

from selenium import webdriver





files = ['ScienceFinancing_year.html', 'NumberOfScientists_year.html', 'NumberOfArticles_year.html']

link = 'file:///home/arleg/PycharmProjects/Courses/Edx/DS/6MidProject/'



# Open a browser

browser = webdriver.Firefox()

# Open each map and save it after loading

for i, file in enumerate(files):

    browser.get('{}{}'.format(link, file))

    time.sleep(5)

    browser.save_screenshot('{}.png'.format(file[:-5]))



# Close browser

browser.quit()

countries = ['Israel', 'Korea, Rep.', 'Japan', 'Finland', 'Sweden']

plt.figure(figsize=(10, 8))

for country in countries:

    time = sci_fi[sci_fi['CountryName'] == country].sort_values('Year')['Year']

    funding = sci_fi[sci_fi['CountryName'] == country].sort_values('Year')['Value']

    country_label = country

    plt.plot(time, funding, label=country_label, linewidth=2)

plt.gca().set_ylim([0, 5])

plt.grid(True)

plt.title('Course of Science financing')

plt.legend(loc='lower right')

plt.savefig('CourseOfScienceFin.png', bbox_inches='tight')

plt.show()

# Taking mean through years for science financing, ratio of researchers and number of articles for each country

sci_fi_avg = sci_fi[['CountryName', 'CountryCode', 'Value']].groupby(['CountryCode', 'CountryName']).mean()

sci_res_avg = sci_res[['CountryName', 'CountryCode', 'Value']].groupby(['CountryCode', 'CountryName']).mean()

sci_art_avg = sci_art[['CountryName', 'CountryCode', 'Value']].groupby(['CountryCode', 'CountryName']).mean()

# Renaming columns

sci_fi_avg.columns = ['Financing' if x == 'Value' else x for x in sci_fi_avg.columns]

sci_res_avg.columns = ['ResearchersNumber' if x == 'Value' else x for x in sci_res_avg.columns]

sci_art_avg.columns = ['ArticlesNumber' if x == 'Value' else x for x in sci_art_avg.columns]



# Merging into one frame

sci_avg = pd.concat([sci_fi_avg, sci_res_avg, sci_art_avg], axis=1)

# Exclude countries with partial information

sci_avg = sci_avg.dropna()

# Let`s look at some statistics of our set

sci_avg.describe()
# Correlations of mean through years financing, number of researchers and number of articles from 129 countries

sci_avg.corr()
# Just for interest let`s look at world latest indicators

world_filter = data['CountryName'] == 'World'

feature_filter = data['IndicatorName'].str.contains('Research and') | data['IndicatorName'].str.contains('Researchers') |  data['IndicatorName'].str.contains('Scientific')

world = data[world_filter & feature_filter]
world = world[world['Year'] == world.groupby('IndicatorName')['Year'].transform(max)]

world
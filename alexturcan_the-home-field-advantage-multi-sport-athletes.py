!pip install upsetplot
import numpy as np

import pandas as pd

pd.set_option('expand_frame_repr', False)

pd.set_option('max_colwidth', 140)

pd.options.mode.chained_assignment = None

import matplotlib.pyplot as plt

import upsetplot

from plotnine import *

from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="specify_your_app_name_here")

from geopy.extra.rate_limiter import RateLimiter

geocode = RateLimiter(geolocator.geocode, min_delay_seconds=2)

import scipy as sp



# ignore MatplotLibDeprecation warning: Passing one of 'on', 'true', 'off', 'false' as a boolean is deprecated.

import warnings

import matplotlib.cbook

warnings.filterwarnings("ignore", category = matplotlib.cbook.mplDeprecation)
olympic_data = pd.read_csv("../input/athlete_events.csv")

print(olympic_data.shape)

print(olympic_data.describe())

olympic_data.head()
event_and_year = olympic_data[['Year', 'Season']].drop_duplicates().sort_values('Year')

(ggplot(event_and_year, aes(x = 'Year', y = 'Season', colour = 'Season')) +

    geom_line() +

    geom_point() +

    scale_x_continuous(breaks = range(event_and_year['Year'].min(), event_and_year['Year'].max()+1, 4)) +

    labs(title = "Timeline of Summer and Winter Olympic Games") +

    theme(plot_title = element_text(size = 10, face = "bold"),

          axis_text_x = element_text(angle = 45),

          axis_text = element_text(size = 7),

          legend_position = "none"))
# Keep distinct pairs of athlete IDs and Season they participated in

athlete_and_season = olympic_data[['ID', 'Season']].drop_duplicates()

# Convert data frame to wide format

athlete_and_season['Participated'] = True

athlete_and_season_wide = athlete_and_season.pivot(index='ID', columns='Season', values='Participated').fillna(False)

# Construct the series for UpSet plotting

columns = list(athlete_and_season_wide.columns)

athlete_and_season_upset = athlete_and_season_wide.groupby(columns).size()



# UpSet plot

upsetplot.plot(athlete_and_season_upset, sort_by = "cardinality") 

plt.suptitle('Athlete participation in seasonal events')

plt.show() 



# Get the summary data behind the UpSet plot

athlete_and_season_summary = athlete_and_season_wide.groupby(columns).size().reset_index(name='Num athletes').sort_values('Num athletes',  ascending=False)

num_athletes = athlete_and_season.ID.nunique()

athlete_and_season_summary['All athletes'] = num_athletes

athlete_and_season_summary['% of all athletes'] = athlete_and_season_summary['Num athletes'] * 100 / athlete_and_season_summary['All athletes']

print(athlete_and_season_summary.to_string(index=False))
# Get the raw data on athletes who participated in both seasons only

athlete_and_season_wide = athlete_and_season_wide.reset_index()

double_season_athletes = athlete_and_season_wide[(athlete_and_season_wide['Summer']==True) & (athlete_and_season_wide['Winter']==True)][['ID']]

double_season_data = pd.merge(double_season_athletes, olympic_data, on = 'ID')

# Keep distinct pairs of athlete IDs and the sports they competed in

double_season_sport = double_season_data[['ID', 'Sport']].drop_duplicates()

# Convert data frame to wide format

double_season_sport['Participated'] = True

double_season_sport_wide = double_season_sport.pivot(index='ID', columns='Sport', values='Participated').fillna(False)

# Construct the series for UpSet plotting

columns = list(double_season_sport_wide.columns)

double_season_sport_upset = double_season_sport_wide.groupby(columns).size()



# UpSet plot

upsetplot.plot(double_season_sport_upset, sort_by = "cardinality", show_counts='%d') 

plt.suptitle('Sports of multi-season athletes')

plt.show() 
sports_both_seasons = olympic_data.groupby('Sport').filter(lambda x: x['Season'].nunique() == 2)

sports_both_seasons['Sport'].unique()
# Get a data set where a medal was won

medal_data = olympic_data[olympic_data.Medal.notnull()]

# Get a list of athlete IDs who won medals in multiple sports

multi_medal_data = medal_data.groupby('ID').filter(lambda x: x['Sport'].nunique() > 1)

multi_medal_athletes = multi_medal_data['ID'].unique()

multi_medal_athletes_df = pd.DataFrame({'ID':multi_medal_athletes})

# Get raw data only on athletes who won medals in multiple sports

multi_medal_sports_data = pd.merge(medal_data, multi_medal_athletes_df, on = 'ID')

print(str(multi_medal_sports_data.ID.nunique()) + ' athletes won medals in multiple sports.')

# Keep distinct pairs of athlete IDs and the sports they won medals in

multi_medal_sports = multi_medal_sports_data[['ID', 'Sport']].drop_duplicates()

# Convert data frame to wide format

multi_medal_sports['Won'] = True

multi_medal_sports_wide = multi_medal_sports.pivot(index='ID', columns='Sport', values='Won').fillna(False)

# Construct the series for UpSet plotting

columns = list(multi_medal_sports_wide.columns)

multi_medal_sports_upset = multi_medal_sports_wide.groupby(columns).size()



# UpSet plot

upsetplot.plot(multi_medal_sports_upset, sort_by = "cardinality", show_counts='%d')

plt.suptitle('Sports of multi-sport medallists')

plt.show() 
# Function that returns the data on athletes who won specific combinations of sports

def get_multi_medal_athlete_data(sports):

  sports_data = medal_data[medal_data.Sport.isin(sports)]

  sports_data_athletes = sports_data.groupby('ID').filter(lambda x: x['Sport'].nunique() == 2)

  sports_data_athlete = sports_data_athletes['ID'].unique()

  return(medal_data[medal_data.ID.isin(sports_data_athlete)])
get_multi_medal_athlete_data(['Shooting', 'Gymnastics'])
get_multi_medal_athlete_data(['Bobsleigh', 'Boxing'])
get_multi_medal_athlete_data(['Handball', 'Swimming'])
# Fix known bugs in the noc_regions dataset

regions = pd.read_csv("../input/noc_regions.csv")

regions.loc[regions['NOC'] == 'BOL', 'region'] = 'Bolivia'

regions.loc[regions['region'] == 'Singapore', 'NOC'] = 'SGP'

# Add the region name into the athlete_events dataset

olympic_data_with_region = pd.merge(olympic_data, regions, on = 'NOC')

print(olympic_data_with_region.shape)

# Are there any NOCs that do not have a corresponding region?

print(olympic_data_with_region.isnull().sum())

# There are 21 events that do not have a region - let's see which NOCs those are.

print(olympic_data_with_region[olympic_data_with_region.region.isnull()]['NOC'].unique())

# 3 NOCs don't have a corresponding region.

# ROT = Refugee Olympic Team

# UNK = Unknown

# TUV = Tuvalu

# All these 3 "regions" are mentioned in the "notes" columns. So we can coalesce the "region" and "notes" columns.

olympic_data_with_region['region'] = olympic_data_with_region.region.combine_first(olympic_data_with_region.notes)
# For every Olympic Games edition, get the number of athletes brought by each country

num_athletes = olympic_data_with_region.groupby(['Games', 'region'])['ID'].nunique().reset_index(name='num_athletes')

print(num_athletes.shape)



# For every Olympic Games edition, get the number of medals won by each country

# Create a unique_medal field which is a concatenation of the year+season+event+medal

medal_data_with_region = olympic_data_with_region[olympic_data_with_region.Medal.notnull()]

medal_data_with_region['unique_medal'] = medal_data_with_region.Games.astype(str) + medal_data_with_region.Event.astype(str) + medal_data_with_region.Medal.astype(str)

num_medals = medal_data_with_region.groupby(['Games', 'region'])['unique_medal'].nunique().reset_index(name='num_medals')

print(num_medals.shape)



# Merge into a single data frame

# If a country didn't win any medals, list 0.

event_summary = pd.merge(num_athletes, num_medals, on = ['Games', 'region'], how = 'left').fillna(0)



# Let's visualise the relationship between number of participating athletes and number of medals won

(ggplot(event_summary, aes(x = 'num_athletes', y = 'num_medals')) +

    geom_point() +

    geom_smooth(method='lm')+

    labs(x = "Number of participating athletes", y = "Number of medals won",

         title = "Relationship between number of athletes a country brings to a game and number of medals won"))
sp.stats.levene(event_summary['num_athletes'], event_summary['num_medals'])
# What do the distributions of the 2 variables look like?

plt.subplot(1, 2, 1)

plt.hist(event_summary['num_athletes'], bins=30)

plt.xlabel('num_athletes')

plt.subplot(1, 2, 2)

plt.hist(event_summary['num_medals'], bins=30)

plt.xlabel('num_medals')

plt.tight_layout()

plt.show()
# Spearman's rank-order correlation

coef, p = sp.stats.spearmanr(event_summary['num_athletes'], event_summary['num_medals'])

print('Spearmans correlation coefficient: %.3f' % coef)

# interpret the significance

alpha = 0.05

if p > alpha:

    print('Number of athletes and medals are uncorrelated (fail to reject H0) p=%.3f' % p)

else:

    print('Number of athletes and medals are correlated (reject H0) p=%.3f' % p)
# Get a list of cities that hosted the Olympic Games and the countries they are in.

cities_df = pd.DataFrame({'City': olympic_data['City'].unique()})

cities_df['location'] = cities_df['City'].apply(geocode, language='en')

cities_df['point'] = cities_df['location'].apply(lambda loc: tuple(loc.point) if loc else None)

cities_df['Country'] = cities_df['point'].apply(lambda point: geolocator.reverse(point, language = 'en').raw['address']['country'])

host_loc = cities_df[['City', 'Country']]

host_loc.head()
# Get the NOC of the host country (to make sure the country names are spelled the same as the "region")

host_loc_with_noc = pd.merge(host_loc, regions, left_on = 'Country', right_on = 'region', how = 'left')

print(host_loc_with_noc[host_loc_with_noc.region.isnull()])

# Turns out that United Kingdom, PRC, and B&H are spelled differently in the noc_regions dataset. We must correct these to continue with the analysis.

# United Kingdom is spelled UK

# PRC is spelled China

# B&H is spelled Bosnia and Herzegovina

host_loc.loc[host_loc['Country'] == 'United Kingdom', 'Country'] = 'UK'

host_loc.loc[host_loc['Country'] == 'PRC', 'Country'] = 'China'

host_loc.loc[host_loc['Country'] == 'B&H', 'Country'] = 'Bosnia and Herzegovina'

# Re-run merge after data was fixed

host_loc_with_noc = pd.merge(host_loc, regions, left_on = 'Country', right_on = 'region', how = 'left')

print(host_loc_with_noc[host_loc_with_noc.region.isnull()])
olympic_data_with_host = pd.merge(olympic_data_with_region, host_loc, on = 'City')

print(olympic_data_with_host.shape)

olympic_data_with_host.head()
olympic_data_with_host['unique_medal'] = olympic_data_with_host.Games.astype(str) + olympic_data_with_host.Event.astype(str) + olympic_data_with_host.Medal.astype(str)





# For every country, find out the year when they hosted

host_year = olympic_data_with_host[['Country', 'Year', 'Season']].drop_duplicates().sort_values(['Season', 'Year'])



# For every host country, find out how many medals they won per athlete when they hosted

host_medals = olympic_data_with_host[(olympic_data_with_host.region == olympic_data_with_host.Country) & olympic_data_with_host.Medal.notnull()].groupby(['region', 'Year', 'Season'])['unique_medal'].nunique().reset_index(name='num_medals')

host_athletes = olympic_data_with_host[(olympic_data_with_host.region == olympic_data_with_host.Country)].groupby(['region', 'Year', 'Season'])['ID'].nunique().reset_index(name='num_athletes')

host_summary = pd.merge(host_athletes, host_medals, on = ['region', 'Year', 'Season'])

host_summary['mpa'] = host_summary['num_medals'] / host_summary['num_athletes']

host_summary = host_summary[['region', 'Year', 'Season', 'mpa']]

host_summary.columns = ['Country', 'Year', 'Season', 'host_mpa']



# Merge the two datasets

host_data = pd.merge(host_year, host_summary, on = (['Country', 'Year', 'Season']))

print(host_data.head())



# For every host country, find out how many medals they won per athlete when they did not host

non_host_medals = olympic_data_with_host[(olympic_data_with_host.region != olympic_data_with_host.Country) & olympic_data_with_host.Medal.notnull()].groupby(['region', 'Year', 'Season'])['unique_medal'].nunique().reset_index(name='num_medals')

non_host_athletes = olympic_data_with_host[(olympic_data_with_host.region != olympic_data_with_host.Country)].groupby(['region', 'Year', 'Season'])['ID'].nunique().reset_index(name='num_athletes')

non_host_summary = pd.merge(non_host_athletes, non_host_medals, on = ['region', 'Year', 'Season'])

non_host_summary['mpa'] = non_host_summary['num_medals'] / non_host_summary['num_athletes']

non_host_summary = non_host_summary[['region', 'Year', 'Season', 'mpa']]

non_host_summary.columns = ['Country', 'Year', 'Season', 'nonhost_mpa']

# Only keep data for countries that were hosts at some point

host_countries = host_summary['Country'].drop_duplicates()

host_countries_df = pd.DataFrame({'Country':host_countries})

non_host_summary = non_host_summary[non_host_summary.Country.isin(host_countries_df['Country'])]

non_host_summary.head()
# ECDF for a single country - Finland

fin = non_host_summary[non_host_summary.Country == 'Finland']

x = np.sort(fin['nonhost_mpa'])

y = np.arange(1, len(x)+1) / len(x)

plt.plot(x, y, marker = '.', linestyle = 'none')

plt.xlabel('Number of medals per athlete')

plt.ylabel('% of olympic games')

plt.margins(0.02) # keep data off plot edges

plt.show()
# For every country, sort the number of medals ascending

non_host_medals_raw_sorted = non_host_summary[['Country', 'nonhost_mpa']].sort_values(['Country', 'nonhost_mpa'])

# For every country, get the number of entries (i.e. number of olympic games in which they participated and won)

lenx = non_host_medals_raw_sorted.groupby('Country').size().reset_index(name = "lenx")

# For every country, compute the ECDF

ecdf_for_all = pd.merge(non_host_medals_raw_sorted, lenx, on = 'Country')

ecdf_for_all['row_number'] = ecdf_for_all.groupby(['Country', 'lenx']).cumcount()+1

ecdf_for_all['ecdf'] = ecdf_for_all['row_number'] / ecdf_for_all['lenx']



(ggplot(ecdf_for_all, aes(x = 'nonhost_mpa', y = 'ecdf', colour = 'Country')) +

    geom_point(stat = 'identity') +

    labs(title = 'ECDF plot of number of medals per athlete won by countries in years when they did not host',

         x = 'Number of medals per athlete',

         y = '% of olympic games') +

    scale_colour_discrete(name = 'Country'))
# Function to get the 90th percentile 

# If the country's ecdf doesn't contain 0.9 we use interpolation to get the mpa value at 0.9 ecdf

def my_interp(x, a=0.9):

    X = np.sort(x)

    e_cdf = np.arange(1, len(X)+1) / len(X)

    if a in e_cdf:

        s = pd.Series(X, index=e_cdf)

        res = s[a]

    else:

        X = np.append(X, np.nan)

        e_cdf = np.append(e_cdf, a)

        s = pd.Series(X, index=e_cdf)

        inter = s.interpolate(method='index')

        res = inter[a]

    return(res)



df = pd.DataFrame(columns = ['Country', 'mpa_percentile90'])

for country in host_countries:

    c = ecdf_for_all[ecdf_for_all.Country == country]

    d = {'Country': [country], 'mpa_percentile90': [my_interp(c['nonhost_mpa'])]}

    res = pd.DataFrame(data=d)

    df = df.append(res)



# Final

final = pd.merge(host_summary, df, on = 'Country')

final['change'] = final['host_mpa'] - final['mpa_percentile90']

final.sort_values('change', ascending=False)
# For every country, get a list of the sports in which they won medals, and the number of medals won

medal_data_with_region = olympic_data_with_region[olympic_data_with_region.Medal.notnull()]

# Create a unique_medal field which is a concatenation of the year+season+event+medal

medal_data_with_region['unique_medal'] = medal_data_with_region.Games.astype(str) + medal_data_with_region.Event.astype(str) + medal_data_with_region.Medal.astype(str)

medal_data_with_region.head()

country_medals = medal_data_with_region.groupby(['region', 'Sport'])['unique_medal'].nunique().reset_index(name='Number of medals').sort_values(['region', 'Number of medals'], ascending=[True, False])

# For every country, get the sport at which they won the most medals

country_top_sport = country_medals.groupby('region').first().reset_index()

country_top_sport.head()
# Get the number and list of countries that won most medals in each sport

sport_top_countries = country_top_sport.groupby('Sport').size().reset_index(name = 'Num countries').sort_values('Num countries')

list_of_countries_per_sport = country_top_sport.groupby('Sport')['region'].apply(np.unique).to_frame()

sport_top_countries_with_list = pd.merge(sport_top_countries, list_of_countries_per_sport, on = 'Sport')

sport_top_countries_with_list.columns = ['Sport', 'Num countries best at it', 'Countries who won most medals in this sport']

sport_top_countries_with_list
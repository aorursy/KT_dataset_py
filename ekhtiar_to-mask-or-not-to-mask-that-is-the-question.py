import json

import pandas as pd

import numpy as np

from datetime import datetime, timedelta

from os import path, listdir, walk

from tqdm.notebook import tqdm

import plotly.express as px

import plotly.graph_objects as go

from sklearn import manifold
from nltk.tokenize import WordPunctTokenizer

from collections import Counter

punct_tokenizer = WordPunctTokenizer()
# read datasets

covid_data_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

# in the covid_19_data.csv we have the number of confirmed cases, deaths, and recovered for every date by country and privince or state.

covid_data_df.head()
covid_measures_df = pd.read_csv('/kaggle/input/covid19-containment-and-mitigation-measures/COVID 19 Containment measures data.csv')

# In the measures dataframe we have a description of the measure implemented and a keyword to standardize all the topics in description. 

# This dataset also has a start date of the measure, and it is given by country and state or province.

covid_measures_df.head()
country_info_df = pd.read_csv('/kaggle/input/countryinfo/covid19countryinfo.csv')

country_info_df.head()
# clean up the data a little bit

country_info_df.loc[country_info_df['alpha2code']=='TW', ['country']] = 'Taiwan'

country_info_df.loc[country_info_df['alpha2code']=='KR', ['country']] = 'South Korea'

country_info_df.loc[country_info_df['alpha2code']=='HK', ['country']] = 'Hong Kong'

country_info_df.loc[country_info_df['alpha2code']=='HK', ['region']] = None

# keep important parameters

country_info_df = country_info_df[country_info_df['region'].isnull()][['country', 'alpha2code', 'pop', 'density', 'medianage', 'urbanpop', 'tests', 'testpop']]
# convert population to million

country_info_df['pop_mil'] = country_info_df['pop'].str.replace(',', '').astype(int) / 1_000_000

country_info_df = country_info_df.drop('pop', axis=1)

# convert date columns to Datetime format

covid_measures_df['Date Start'] = pd.to_datetime(covid_measures_df['Date Start'], format="%b %d, %Y")

covid_data_df['ObservationDate'] = pd.to_datetime(covid_data_df['ObservationDate'], format="%m/%d/%Y")

# Some cleaning on country names

country_info_df['country'] = country_info_df['country'].apply(lambda row: str(row)).apply(lambda row: 'Czech Republic' if 'Czechia' in row else row)

covid_measures_df['Country'] = covid_measures_df['Country'].apply(lambda row: str(row)).apply(lambda row: 'Czech Republic' if 'Czechia' in row else row)

covid_measures_df['Country'] = covid_measures_df['Country'].apply(lambda row: str(row)).apply(lambda row: 'United States' if 'US' in row else row)
# see number of total examples found found and missing measures against coronavirus 

# As you can see, we have a few rows where measure and their date is missing

print(f"total examples {len(covid_measures_df)}")

print(f"measures description found: {len(covid_measures_df[covid_measures_df['Description of measure implemented'].notnull()])}")

print(f"measures keywords found: {len(covid_measures_df[covid_measures_df['Keywords'].notnull()])}")

print(f"measures with date found: {len(covid_measures_df[covid_measures_df['Date Start'].notnull()])}")
def get_measures_count_by_country(country, covid_data_df, covid_measures_df, country_info_df):

    '''

    Get a datafarme containing number of covid cases and measure takens in a single dataframe

    '''

    country_covid_df = covid_data_df[covid_data_df['Country/Region'] == country].groupby(['Country/Region', 'ObservationDate']).sum().reset_index()

    country_measures_df = covid_measures_df[covid_measures_df['Country']==country]

    country_covid_df['Confirmed Increase'] = country_covid_df['Confirmed'].diff().fillna(0)

    country_covid_df['Death Increase'] = country_covid_df['Deaths'].diff().fillna(0)

    country_covid_df['Recovered Increase'] = country_covid_df['Recovered'].diff().fillna(0)

    country_df = country_covid_df.merge(country_measures_df, how='left',left_on='ObservationDate', right_on='Date Start')

    pop_mil = country_info_df[country_info_df['country']==country]['pop_mil']

    pop_mil = int(pop_mil)

    country_df['confirmed_per_one_mil'] = country_df['Confirmed'] / pop_mil

    country_df['death_per_one_mil'] = country_df['Deaths'] / pop_mil

    country_df['recovered_per_one_mil'] = country_df['Recovered'] / pop_mil

    return country_df
def insert_breaks(measure):

    '''

    The hover textbox is going to become very long. A work around to this is to insert <br> tags and break the lines in text.

    '''

    measure_list = str(measure).split(' ')

    [measure_list.insert(x, '<br>') for x in range(0, len(measure_list), 7)]

    measure_list.pop(0)

    return ' '.join(measure_list)



def get_count_list(country_df):

    country_df = country_df.sort_values('ObservationDate')

    country_df = country_df[['ObservationDate','Confirmed','Deaths']].drop_duplicates()

    return (country_df['Confirmed'].to_list(), country_df['Deaths'].to_list())



def get_measures(country_df):

    '''

    get x y co-ordinates and measures and keywords

    '''

    country_df = country_df.sort_values('ObservationDate')

    x = country_df[country_df['Description of measure implemented'].notnull()]['Date Start'].to_list()

    y = country_df[country_df['Description of measure implemented'].notnull()]['Confirmed'].to_list()

    measures = country_df[country_df['Description of measure implemented'].notnull()]['Description of measure implemented'].to_list()

    measures = [insert_breaks(measure) for measure in measures]

    about_mask = country_df[country_df['Description of measure implemented'].notnull()]['mask'].to_list()

    about_mask = ['😷' if event is True else '' for event in about_mask]

    return (x, y, measures, about_mask)
def plot_measures(country_df, x, y, measures, keywords=None):

    

    '''

    plots the number of confirmed cases in line graph and measures using scatter plots

    '''



    fig = px.line(country_df, x="ObservationDate", y="Confirmed", color="Country/Region", line_group="Country/Region")

    

    #fig.add_trace(go.Bar(

    #    x=country_df['ObservationDate'], 

    #    y=country_df['Confirmed Increase']))

    

    country = country_df['Country/Region'][0]

    

    # taking care of events with same date

    

    # create an empty list to store dates with multiple events

    dates_with_multiple_events = []

    

    # store dates with multiple events in the dataset

    for dt in set(x):

        if len([t for t in x if t==dt]) > 1:

            dates_with_multiple_events.append(dt)

    

    # loop over x and add a few hours to make each date different 

    x = [t+timedelta(minutes=i+1) if t in dates_with_multiple_events else t for i, t in enumerate(x)]

    #print(x)

    

    fig.add_trace(go.Scatter(

        x=x, y=y, text=measures,

        mode="markers",

        name="measures"

    ))

    

    if keywords:

        fig.add_trace(go.Scatter(

            x=x, y=y, text=keywords,

            textposition='top left',

            mode="text",

            name="keywords",

            hoverinfo='skip'

        ))

    

    fig.update_layout(hovermode='closest', showlegend=False, 

                     title = 'Covid 19 Measures vs Confirmed Cases - '+country + ' (Hover To See Measure)',

                     xaxis_title = 'Date', yaxis_title = 'Confirmed Covid-19 Cases')

    

    return fig

def check_for_mask(text):

    word_list = ['mask', 'masks']

    text = str(text)

    token = set(punct_tokenizer.tokenize(text.lower()))

    match = set(word_list).intersection(token)

    if match:

        return True

    else:

        return False
# check for masks in description and create a column to indicate so

covid_measures_df['mask'] = covid_measures_df['Description of measure implemented'].apply(check_for_mask)
# count number of mask related events for each country and do a bar plot

mask_country_count = covid_measures_df[covid_measures_df['mask']==True]['Country'].value_counts()

fig = px.bar(mask_country_count, x = mask_country_count.index, y = mask_country_count.values)

fig.update_layout(title = 'Countries with mask mentioned in measures dataset',

                  xaxis_title = 'Country Name', yaxis_title = 'Number of events in data')

fig.show()
country_info_df[country_info_df['country']=='South Korea']
kr_df = get_measures_count_by_country('South Korea', covid_data_df, covid_measures_df, country_info_df)

cz_df = get_measures_count_by_country('Czech Republic', covid_data_df, covid_measures_df, country_info_df)

sg_df = get_measures_count_by_country('Singapore', covid_data_df, covid_measures_df, country_info_df)

hk_df = get_measures_count_by_country('Hong Kong', covid_data_df, covid_measures_df, country_info_df)

sk_df = get_measures_count_by_country('Slovakia', covid_data_df, covid_measures_df, country_info_df)

tw_df = get_measures_count_by_country('Taiwan', covid_data_df, covid_measures_df, country_info_df)

mask_country_df = pd.concat([kr_df, cz_df, sg_df, hk_df, sk_df, tw_df])
# plot using plotly

fig = px.line(mask_country_df, x="ObservationDate", y="confirmed_per_one_mil", color="Country/Region", line_group="Country/Region")

fig.update_layout(title = 'Covid 19 Confirmed Cases for Countries Where Everyone Wears A Mask',

                  xaxis_title = 'Date', yaxis_title = 'Confirmed Covid-19 Cases (Per 1 mil)')

fig.show()
x, y, measures, mask_events = get_measures(kr_df)

plot_measures(kr_df, x, y, measures, mask_events)
x, y, measures, mask_events = get_measures(cz_df)

plot_measures(cz_df, x, y, measures, mask_events)
x, y, measures, mask_events = get_measures(sg_df)

plot_measures(sg_df, x, y, measures, mask_events)
x, y, measures, mask_events = get_measures(hk_df)

plot_measures(hk_df, x, y, measures, mask_events)
x, y, measures, mask_events = get_measures(sk_df)

plot_measures(sk_df, x, y, measures, mask_events)
x, y, measures, mask_events = get_measures(tw_df)

plot_measures(tw_df, x, y, measures, mask_events)
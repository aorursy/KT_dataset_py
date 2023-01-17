import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns



import plotly.plotly as py

import plotly.graph_objs as go 

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.figure_factory as ff



import urllib.request

import json

import numbers



sns.set_style('whitegrid')

%matplotlib inline

init_notebook_mode(connected=True) 
df = pd.read_csv('../input/201709301651_masters_portal.csv',

                 low_memory=False,

                 na_values=['NaN', 'nan'],

                 na_filter=True)
df.info()
df.head()
top_countries = df['country_name'].value_counts().sort_values(ascending=False)

top_countries.head(20)
top_countries.describe()
top_countries_norm = (top_countries / top_countries.sum()) * 100

top_countries_norm.head(20)
top_countries_norm.head(20).sum()
countries_data = df.groupby(['country_name', 'country_code'], as_index=False).count()

countries_data.head()
data = dict(type = 'choropleth',

            locations = countries_data['country_code'],

            colorscale= 'Viridis',

            text= countries_data['country_name'],

            z=countries_data['program_url'],

            colorbar = {'title':'No. of programs'})

layout = dict(

    title = 'Master Degrees By country 2017',

    geo = dict(

        showframe = False,

        projection = {'type':'Mercator'}

    )

)

choromap = go.Figure(data = [data],layout = layout)

iplot(choromap)
program_types = df['program_type'].value_counts().sort_values(ascending=False)

program_types.head(20)
data = [

    {

        'y': list(program_types[:30]),

        'x': program_types.index[:30],

        'mode': 'markers',

        'marker': {

            'color': program_types,

            'size': 20,

            'showscale': True

        }

    }

]



iplot(data)
program_names = df['program_name'].value_counts().sort_values(ascending=False)

program_names.head(50)
len(program_names)
df['tution_1_currency'].unique()
# ecxhange_url = 'http://api.fixer.io/latest?base=USD'

# exchange_rates = urllib.request.urlopen(ecxhange_url).read().decode('utf8')

# exchange_rates = json.loads(exchange_rates)

exchange_rates = {'base': 'USD',

 'date': '2017-09-29',

 'rates': {'AUD': 1.2769,

  'BGN': 1.6566,

  'BRL': 3.1878,

  'CAD': 1.244,

  'CHF': 0.97044,

  'CNY': 6.652,

  'CZK': 22.007,

  'DKK': 6.3038,

  'EUR': 0.84703,

  'GBP': 0.74689,

  'HKD': 7.8108,

  'HRK': 6.3485,

  'HUF': 263.15,

  'IDR': 13458.0,

  'ILS': 3.5229,

  'INR': 65.28,

  'JPY': 112.5,

  'KRW': 1145.0,

  'MXN': 18.178,

  'MYR': 4.2205,

  'NOK': 7.9726,

  'NZD': 1.3852,

  'PHP': 50.864,

  'PLN': 3.6458,

  'RON': 3.8957,

  'RUB': 57.811,

  'SEK': 8.173,

  'SGD': 1.3579,

  'THB': 33.32,

  'TRY': 3.5586,

  'ZAR': 13.505}}
df['tution_1_type'].unique()
df['tution_2_type'].unique()
def convert_to_usd(amount=None, currency=None):

    if not np.isnan(amount) and isinstance(amount, numbers.Number):

        if currency == 'USD':

            rate = 1

        elif currency == 'Free':

            rate = 0

        else:

            rate = exchange_rates['rates'][currency]

        usd = amount * rate

        return usd

    else:

        return amount

    



df['tuition_1_USD'] = df.apply(lambda x: convert_to_usd(x['tution_1_money'], x['tution_1_currency']), axis=1)
international_tuition = df[df['tution_1_type'] == 'International']['tuition_1_USD']
plt.figure(figsize=(12,7))



sns.boxplot(international_tuition, orient='v', width=0.5)



plt.ylabel("Tuition", fontsize=18)

plt.title("International Tuition", fontsize=18)
df['tuition_1_cat'] = pd.cut(international_tuition,

                     [i for i in range(0, 14000, 1000)],

                     labels=[i for i in range(0, 13000, 1000)])
plt.figure(figsize=(12,7))



sns.countplot(df['tuition_1_cat'])



plt.xticks(rotation=45)

plt.xlabel("Tuition in USD", fontsize=18)

plt.ylabel("Count", fontsize=18)

plt.title("International Tuition Distribution in USD", fontsize=20)
df['university_rank_cat'] = pd.cut(df['university_rank'],

                     [i for i in range(0, 900, 50)],

                     labels=[i for i in range(50, 900, 50)])
plt.figure(figsize=(12,7))



sns.countplot(df['university_rank_cat'])



plt.xticks(rotation=45)

plt.xlabel("Uneversity Ranks", fontsize=18)

plt.ylabel("Count", fontsize=18)

plt.title("Uneversity Ranks Distribution", fontsize=18)
df['duration'].unique()
def convert_duration_to_months(duration):

    if isinstance(duration, numbers.Number):

        return duration

    if (duration.find('months') > -1):

        return int(duration[:duration.find('months') - 1])

    if (duration.find('days') > -1):

        return int(int(duration[:duration.find('days') - 1]) / 30)

    return np.nan



df['duration_months'] = df['duration'].apply(lambda x: convert_duration_to_months(x))

df['duration_months_cat'] = pd.cut(df['duration_months'],

                     [i for i in range(0, 100, 3)],

                     labels=[i for i in range(3, 100, 3)])

plt.figure(figsize=(12,7))



sns.countplot(df['duration_months_cat'])



plt.xticks(rotation=45)

plt.xlabel("Uneversity Ranks", fontsize=18)

plt.ylabel("Count", fontsize=18)

plt.title("Uneversity Ranks Distribution", fontsize=18)
computer_science_filtered = df[df['program_name'].map(lambda x : x.lower().find('computer') > -1

                                                 or x.lower().find('software') > -1

                                                 or x.lower().find('data') > -1 )]

computer_science_programs_counts =  computer_science_filtered['program_name'].value_counts().sort_values(ascending=False)

computer_science_programs_counts.head(20)
sum(computer_science_programs_counts)
computer_science_filtered.head()
computer_science_filtered.sort_values(by='university_rank').head(20)
computer_science_filtered['ielts_score'].describe()
sns.boxplot(y=computer_science_filtered['ielts_score'])
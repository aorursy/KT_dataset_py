from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))

%reload_ext autoreload

%autoreload 2

%matplotlib inline

# <font color=blue>Text</font>
import sys

from os.path import join, splitext, isfile, basename, getsize

from os import listdir

from tqdm import tqdm_notebook as tqdm

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from datetime import datetime

from collections import defaultdict

import pandas as pd

import pycountry

import plotly.offline as py

import warnings

import cufflinks as cf

cf.go_offline()

warnings.filterwarnings('ignore')

sns.set_style('darkgrid')

data = '../input'
def get_csv_from_folder(input_folder):

    """

    input:  directory path

    output: list with all the csv's full paths from given path

    """

    return [join(input_folder, f) for f in listdir(input_folder) if

            isfile(join(input_folder, f)) and splitext(f)[1] == '.csv']
docs = get_csv_from_folder(data); docs
data_dict = defaultdict()

for doc in tqdm(docs):

    data_dict[basename(splitext(doc)[0])] = pd.read_csv(doc, low_memory=False)
for key in data_dict.keys():

    print(key)

    print(data_dict[key].info())

    print('----------------------------------------')
data_dict['clicks_train'].head()
data_dict['clicks_test'].head()
train = data_dict['clicks_train'].groupby('display_id')['ad_id'].count().value_counts(); train.head()
test = data_dict['clicks_test'].groupby('display_id')['ad_id'].count().value_counts(); test.head()
fig, (first_axis, second_axis) = plt.subplots(1, 2, figsize=(15, 10))



train_plot = train.plot.pie(ax=first_axis, cmap = 'winter', fontsize=10)

test_plot = test.plot.pie(ax=second_axis, cmap = 'summer', fontsize=10)
#frequency of every ad (first column -> lines, second column -> frequency)

ads_frequency = data_dict['clicks_train'].groupby('ad_id').ad_id.count().sort_values(); ads_frequency.head()
ads_percentage = pd.Series()

for i in [2**x for x in range(1, 10)]:

    ads_percentage[str(i)] =  round((ads_frequency < i).mean() * 100, 2)
#let's plot both the frequency and the percentage for viewed adds

fig, (first_axis, second_axis) = plt.subplots(1, 2, figsize=(30, 5))



frequency_plot = sns.boxplot([ads_frequency], ax=first_axis)

frequency_plot = frequency_plot.set(xlabel='Frequency')

percentage_plot = ads_percentage.plot(kind ='bar', ax=second_axis, cmap = 'summer', fontsize=14)

percentage_plot = percentage_plot.set(ylabel='Percentage', xlabel='Less than X times')

for patch in second_axis.patches:

    second_axis.annotate('%{:.2f}'.format(patch.get_height()), (patch.get_x(), patch.get_height()), fontsize=12)
ads_clicked = data_dict['clicks_train'][data_dict['clicks_train'].clicked == 1].groupby('ad_id').ad_id.count().sort_values(); ads_clicked.head()
ads_clicked_percentage = pd.Series()

for i in [2**x for x in range(1, 10)]:

    ads_clicked_percentage[str(i)] =  round((ads_clicked < i).mean() * 100, 2)
#let's plot both the frequency and the percentage for clicked adds

fig, (first_axis, second_axis) = plt.subplots(1, 2, figsize=(30, 5))



ads_clicked_plot = ads_clicked.plot(kind='hist', bins=100, log=True, ax=first_axis, color='indianred', fontsize=14)

ads_clicked_plot = ads_clicked_plot.set(ylabel='log10(frequency)', xlabel='Number of times add was clicked')

ads_clicked_percentage_plot = ads_clicked_percentage.plot(kind ='bar', ax=second_axis, cmap='summer', fontsize=14)

ads_clicked_percentage_plot = ads_clicked_percentage_plot.set(ylabel='Percentage', xlabel='Less than X times')

for patch in second_axis.patches:

    second_axis.annotate('%{:.2f}'.format(patch.get_height()), (patch.get_x(), patch.get_height()), fontsize=12)
fig, (first_axis) = plt.subplots(1, 1, figsize=(30, 5))



ads_clicked.name = 'Frequency for clicks'

ads_frequency.name =  'Frequency for views'



ads_clicked.plot(kind='hist', bins=100, density=True, log=True, cmap='rainbow', legend=True, alpha=0.5, fontsize=14)

ads_frequency.plot(kind='hist', bins=100, density=True, log=True, legend=True, alpha=0.5, fontsize=14)

first_axis.set(ylabel='Log10(Frequency)', xlabel='Count of Clicks');
data_dict['documents_categories'].head()
data_dict['documents_topics'].head()
data_dict['documents_entities'].head()
category_ids = data_dict['documents_categories'].groupby('category_id').confidence_level.count().sort_values(); category_ids.tail()
print('There are {} unique categories'.format(len(category_ids)))
fig, (first_axis) = plt.subplots(1, 1, figsize=(10, 15))



doc_category_plot = category_ids.plot(kind='barh', ax=first_axis, color='black', alpha = 0.8, fontsize=9)

plt.xlabel('Categories', fontsize=14)

plt.ylabel('Occurences', fontsize=14);
topics_ids = data_dict['documents_topics'].groupby('topic_id').confidence_level.count().sort_values(); category_ids.tail()
print('There are {} unique topics'.format(len(topics_ids)))
fig, (first_axis) = plt.subplots(1, 1, figsize=(10, 43))



doc_topic_plot = topics_ids.plot(kind='barh', ax=first_axis, color='magenta', alpha = 0.8, fontsize=9)

plt.xlabel('Topics', fontsize=14)

plt.ylabel('Occurences', fontsize=14);
entity_ids = data_dict['documents_entities'].groupby('entity_id').confidence_level.count().sort_values(); entity_ids.tail()
print('There are {} unique entities'.format(len(entity_ids)))
data_dict['page_views_sample'].head()
locations = data_dict['page_views_sample'].geo_location.apply(lambda geo_info: str(geo_info).split('>')[0]); locations.tail()
country_frequence = locations.value_counts()

country_frequence_precentage = round(country_frequence / country_frequence.sum(), 4); country_frequence_precentage.head()
fig, (first_axis) = plt.subplots(1, figsize=(50, 5))

country_frequence_precentage.plot(kind='bar', color='indianred', ax=first_axis);
top_10_countries = country_frequence_precentage[:10]

fig, (first_axis) = plt.subplots(1, figsize=(15, 5))

top_10_countries.plot.bar(color='indianred', ax=first_axis);

for patch in first_axis.patches:

    first_axis.annotate('%{:.4f}'.format(patch.get_height()), (patch.get_x()- .1, patch.get_height()), fontsize=12)
country_percentage_df = pd.DataFrame({'country':country_frequence_precentage.index, 'value':country_frequence_precentage.values}); country_percentage_df.head()
country_df = pd.DataFrame({'country':country_frequence.index, 'value':country_frequence.values}); country_df.head()
country_code_code3 = {}

country_names = list(pycountry.countries)



for country in country_names:

    country_code_code3[country.alpha_2] = country.name

    

for indx, country in enumerate(country_df['country']):

    if country not in country_code_code3: continue

    country_df['country'][indx] = country_code_code3[country]

    country_percentage_df['country'][indx] = country_code_code3[country]

country_df.head()
data = [ dict(

        type = 'choropleth',

        locations = country_df['country'],

        z = country_df['value'],

        locationmode = 'country names',

        scl = [[0, 'rgb(218,218,235)'], [0.005, 'rgb(117,107,177)'], [1, 'rgb(84,39,143)']],

        marker = dict(

            line = dict(color = 'rgb(0, 0, 0)', width = .8)),

            colorbar = dict(autotick = True, tickprefix = '', title = 'Scale')

            )

       ]



layout = dict(

    title = '#Users by country',

    geo = dict(

        showframe = False,

        showcoastlines = True,

        projection = dict(

        type = 'equirectangular'

        )

            )

    )



fig = dict(data=data, layout=layout)

py.iplot(fig, validate=False, filename='map')
data = [ dict(

        type = 'choropleth',

        locations = country_df['country'],

        z = country_percentage_df['value'],

        locationmode = 'country names',

        scl = [[0, 'rgb(218,218,235)'], [0.005, 'rgb(117,107,177)'], [1, 'rgb(84,39,143)']],

        marker = dict(

            line = dict(color = 'rgb(0, 0, 0)', width = .8)),

            colorbar = dict(autotick = True, tickprefix = '', title = 'Scale')

            )

       ]



layout = dict(

    title = 'Percentage of users by country',

    geo = dict(

        showframe = False,

        showcoastlines = True,

        projection = dict(

        type = 'equirectangular'

        )

            )

    )



fig = dict(data=data, layout=layout)

py.iplot(fig, validate=False, filename='worldmap')
platforms = data_dict['page_views_sample']['platform']; platforms.head()
platforms_frequency = data_dict['page_views_sample'].groupby('platform').platform.count(); platforms_frequency.head()
# platforms_frequency = platforms_frequency.drop(labels=['\\N']); platforms_frequency.head()
fig, (first_axis) = plt.subplots(1, figsize=(10, 10))

platforms_frequency.plot.pie(y='Count', ax=first_axis, cmap='spring', labels=['Desktop (1)', 'Smartphone (2)', 'Tablet (3)'], title='Page Views bases on Platform', fontsize=15, autopct='%.2f');
users = data_dict['page_views_sample'].groupby('uuid').uuid.count(); users.head()
users.sort_values()
users_percentage = pd.Series()

for i in [2**x for x in range(1, 4)]: users_percentage[str(i)] = round((users.values < i).mean() * 100, 2)

users_percentage.head()
fig, (first_axis, second_axis) = plt.subplots(2, 1, figsize=(10, 14))

first_axis.set(xlabel = 'Number of users', ylabel = 'log10(Frequency)')

users.plot.hist(ax = first_axis, color = 'indianred', log = True, bins = 100, fontsize=16)

users_percentage.plot.bar(color = 'indianred', ax = second_axis, fontsize = 16)

second_axis.set(xlabel = 'Users apparence', ylabel = 'percentage')

for patch in second_axis.patches:

    second_axis.annotate('%{:.2f}'.format(patch.get_height()), (patch.get_x() + 0.1, patch.get_height()), fontsize=16)
traffic_counts = data_dict['page_views_sample'].groupby('traffic_source').traffic_source.count(); traffic_counts.head()
fig, (first_axis) = plt.subplots(1, figsize=(10, 10))

traffic_counts.plot.pie(x='Count', ax=first_axis, cmap='summer', labels=['Interior (1)', 'Search (2)', 'Social (3)'], title= 'Page views by Traffic', fontsize=15, autopct='%.2f');
platform_traffic = data_dict['page_views_sample'].groupby(['platform', 'traffic_source']).size(); platform_traffic
fig, (first_axis) = plt.subplots(1, figsize=(10, 10))

platform_traffic.plot.pie(x='Count', ax=first_axis, cmap='summer', labels=['Desktop Interior (1-1)', 'Desktop Search (1-2)', 'Desktop Social (1-3)', 'Smartphone Interior (2-1)', 'Smartphone Search (2-2)', 'Smartphone Social (2-3)', 'Tablet Interior (3-1)', 'Tablet Search (3-2)', 'Tablet Social (3-3)',], title= 'Page views by Platform and Traffic', fontsize=15, autopct='%.2f');
locations = data_dict['events'].geo_location.apply(lambda geo_info: str(geo_info).split('>')[0]); locations.tail()
country_frequence = locations.value_counts()

country_frequence_precentage = round(country_frequence / country_frequence.sum(), 4); country_frequence_precentage.head()
top_10_countries = country_frequence_precentage[:10]

fig, (first_axis) = plt.subplots(1, figsize=(15, 5))

top_10_countries.plot.bar(color='indianred', ax=first_axis);

for patch in first_axis.patches:

    first_axis.annotate('%{:.4f}'.format(patch.get_height()), (patch.get_x()- .1, patch.get_height()), fontsize=12)
platforms = data_dict['events']['platform']; platforms.head()
platforms_frequency = data_dict['events'].groupby('platform').platform.count(); platforms_frequency.head()
platforms_frequency = platforms_frequency.drop(labels=['\\N']); platforms_frequency.head()
fig, (first_axis) = plt.subplots(1, figsize=(10, 10))

platforms_frequency.plot.pie(y='Count', ax=first_axis, cmap='spring', labels=['Desktop (1)', 'Smartphone (2)', 'Tablet (3)'], title='Events based on Platform', fontsize=15, autopct='%.2f');
users = data_dict['events'].groupby('uuid').uuid.count(); users.head()
users_percentage = pd.Series()

for i in [2**x for x in range(1, 4)]: users_percentage[str(i)] = round((users.values < i).mean() * 100, 2)

users_percentage.head()
fig, (first_axis, second_axis) = plt.subplots(2, 1, figsize=(10, 14))

first_axis.set(xlabel = 'Number of users', ylabel = 'log10(Frequency)')

users.plot.hist(ax = first_axis, color = 'indianred', log = True, bins = 100, fontsize=16)

users_percentage.plot.bar(color = 'indianred', ax = second_axis, fontsize = 16)

second_axis.set(xlabel = 'Users apparence', ylabel = 'percentage')

for patch in second_axis.patches:

    second_axis.annotate('%{:.2f}'.format(patch.get_height()), (patch.get_x() + 0.1, patch.get_height()), fontsize=16)
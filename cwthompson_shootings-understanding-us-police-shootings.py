import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

import seaborn as sns

import plotly.graph_objects as go
data = pd.read_csv('../input/police-deadly-force-usage-us/fatal-police-shootings-data.csv')

data.head(5)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))



sns.countplot(x='race', orient='v', ax=ax1, data=data)

sns.countplot(x='gender', orient='v', ax=ax2, data=data)

sns.countplot(x='signs_of_mental_illness', orient='v', ax=ax3, data=data)

sns.countplot(x='age', orient='v', ax=ax4, data=data)

ax4.set_xticks(range(0, 90, 10))

ax4.set_xticklabels(range(0, 90, 10))

fig.suptitle('Police Shooting Victim Data', fontsize=16, y=1.02)

plt.tight_layout()

plt.show()
# Get data

us_census_data = pd.read_csv('../input/us-census-demographic-data/acs2017_county_data.csv')



# Get population proportions

total_population = us_census_data['TotalPop'].sum()

race_proportions = pd.DataFrame(['White', 'Hispanic', 'Black', 'Asian', 'Native'], columns=['Race'])

race_proportions['Population'] = race_proportions['Race'].apply(lambda x: us_census_data.apply(lambda y: y['TotalPop'] * y[x] / total_population, axis=1).sum())

race_proportions['Killed In Police Shootings'] = race_proportions['Race'].apply(lambda x: data[data['race'] == x[0]].shape[0] * 100 / data.shape[0])



# Plot proportions

race_proportions = race_proportions.melt(id_vars='Race')

fig, ax = plt.subplots(1, 1, figsize=(10,6))

sns.barplot(x='value', y='Race', hue='variable',data=race_proportions, ax=ax,

            orient='h', palette=['#0390fc', '#ff3321'])



# Annotate with values

for p in ax.patches:

    width = p.get_width()

    plt.text(3+p.get_width(), p.get_y()+0.55*p.get_height(),

             '{:1.2f}%'.format(width),

             ha='center', va='center')



# Customise and show

ax.set_title('Percentage of deaths from police shootings\ncompared to percentage of population by race', fontsize=16)

ax.tick_params(axis='both', labelsize=12)

for spine in ax.spines.values():

    spine.set_visible(False)

ax.set_xlabel('')

ax.set_ylabel('')

ax.set_xticks([])

plt.legend(frameon=False, fontsize=12, ncol=2)

plt.tight_layout()

plt.show()
state_counts = data.groupby(by='state').agg({'id' : 'count'}).reset_index()



fig = go.Figure(data=go.Choropleth(

    locations=state_counts['state'],

    z = state_counts['id'],

    locationmode = 'USA-states',

    colorscale = 'Reds',

    colorbar_title = "Deaths"

))



fig.update_layout(

    title_text = 'Police Shooting Deaths by US States',

    geo_scope='usa'

)



fig.show()
state_pops = pd.read_csv('../input/us-state-populations-2018/State Populations.csv')

state_codes = {'California' : 'CA', 'Texas' : 'TX', 'Florida' : 'FL', 'New York' : 'NY', 'Pennsylvania' : 'PA',

       'Illinois' : 'IL', 'Ohio' : 'OH', 'Georgia' : 'GA', 'North Carolina' : 'NC', 'Michigan' : 'MI',

       'New Jersey' : 'NJ', 'Virginia' : 'VA', 'Washington' : 'WA', 'Arizona' : 'AZ', 'Massachusetts' : 'MA',

       'Tennessee' : 'TN', 'Indiana' : 'IN', 'Missouri' : 'MO', 'Maryland' : 'MD', 'Wisconsin' : 'WI',

       'Colorado' : 'CO', 'Minnesota' : 'MN', 'South Carolina' : 'SC', 'Alabama' : 'AL', 'Louisiana' : 'LA',

       'Kentucky' : 'KY', 'Oregon' : 'OR', 'Oklahoma' : 'OK', 'Connecticut' : 'CT', 'Iowa' : 'IA', 'Utah' : 'UT',

       'Nevada' : 'NV', 'Arkansas' : 'AR', 'Mississippi' : 'MS', 'Kansas' : 'KS', 'New Mexico' : 'NM',

       'Nebraska' : 'NE', 'West Virginia' : 'WV', 'Idaho' : 'ID', 'Hawaii' : 'HI', 'New Hampshire' : 'NH',

       'Maine' : 'ME', 'Montana' : 'MT', 'Rhode Island' : 'RI', 'Delaware' : 'DE', 'South Dakota' : 'SD',

       'North Dakota' : 'ND', 'Alaska' : 'AK', 'District of Columbia' : 'DC', 'Vermont' : 'VT',

       'Wyoming' : 'WY'}

state_pops['State Codes'] = state_pops['State'].apply(lambda x: state_codes[x])

state_counts['Pop'] = state_counts['state'].apply(lambda x: state_pops[state_pops['State Codes'] == x].reset_index()['2018 Population'][0])



fig = go.Figure(data=go.Choropleth(

    locations=state_counts['state'],

    z = state_counts['id'] / state_counts['Pop'] * 100000,

    locationmode = 'USA-states',

    colorscale = 'Reds',

    colorbar_title = "Deaths Per 100,000"

))



fig.update_layout(

    title_text = 'Police Shooting Deaths by US States per 100,000 People',

    geo_scope='usa'

)



fig.show()
from datetime import date



# Get date month data

data['date'] = pd.to_datetime(data['date'])

newd = data.groupby(pd.Grouper(key='date', freq='M')).count().reset_index()[['date', 'id']]

newd['date_ordinal'] = newd['date'].apply(lambda x: x.toordinal())



# Plot

fig, ax = plt.subplots(1, 1, figsize=(12, 4))

sns.regplot(x='date_ordinal', y='id', ci=95, ax=ax, data=newd)



# Customise

year_labels = [newd['date_ordinal'].min() + (x * 365) for x in range(6)]

ax.set_xticks(year_labels)

ax.set_xticklabels([2015, 2016, 2017, 2018, 2019, 2020])

ax.set_xlabel('Year')

ax.set_ylabel('Deaths')



plt.title('US Police Shooting Deaths Over Time', fontsize=16)

plt.show()
unarmed_weapons = ['unarmed', 'toy weapon', np.nan, 'undetermined', 'flashlight']



data['is_armed'] = data['armed'].apply(lambda x: 'Armed' if x not in unarmed_weapons else 'Unarmed')

unarmed_data = data[data['is_armed'] == 'Unarmed']

armed_data = data[data['is_armed'] == 'Armed']
fig = plt.figure(figsize=(12, 12))

gs=GridSpec(4,2)

ax0 = fig.add_subplot(gs[0,:])

ax1 = fig.add_subplot(gs[1,0])

ax2 = fig.add_subplot(gs[1,1])

ax3 = fig.add_subplot(gs[2,0])

ax4 = fig.add_subplot(gs[2,1])

ax5 = fig.add_subplot(gs[3,0])

ax6 = fig.add_subplot(gs[3,1])



fig.suptitle('Unarmed vs Armed', y=1.03, fontsize=18)



# Unarmed vs Armed

sns.countplot('is_armed', ax=ax0, data=data, order=['Unarmed', 'Armed'])

ax0.set_xlabel('')



# Race

sns.barplot(x='race', y='race', orient='v', ax=ax1, data=unarmed_data,

            estimator=lambda x: len(x) / len(unarmed_data) * 100, order=['W', 'B', 'H', 'A', 'N', 'O'])

sns.barplot(x='race', y='race', orient='v', ax=ax2, data=armed_data,

            estimator=lambda x: len(x) / len(armed_data) * 100, order=['W', 'B', 'H', 'A', 'N', 'O'])

for ax in [ax1, ax2]:

    ax.set_ylabel('percent')



# Gender

sns.barplot(x='gender', y='gender', orient='v', ax=ax3, data=unarmed_data,

            estimator=lambda x: len(x) / len(unarmed_data) * 100, order=['M', 'F'])

sns.barplot(x='gender', y='gender', orient='v', ax=ax4, data=armed_data,

            estimator=lambda x: len(x) / len(armed_data) * 100, order=['M', 'F'])

for ax in [ax3, ax4]:

    ax.set_ylabel('percent')

    ax.set_yticks(range(0, 100, 20))

    ax.set_yticklabels(range(0, 100, 20))



# Age

sns.barplot(x='age', y='age', orient='v', ax=ax5, data=unarmed_data,

            estimator=lambda x: len(x) / len(unarmed_data) * 100)

sns.barplot(x='age', y='age', orient='v', ax=ax6, data=armed_data,

            estimator=lambda x: len(x) / len(armed_data) * 100)

for ax in [ax5, ax6]:

    ax.set_ylabel('percent')

    ax.set_xticks(range(0, 90, 10))

    ax.set_xticklabels(range(0, 90, 10))

    ax.set_yticks(range(0, 5))

    ax.set_yticklabels(range(0, 5))



plt.tight_layout()

plt.show()
from scipy.stats import chi2_contingency



unarmed_vals = unarmed_data.groupby('race').count()['id'].reset_index().sort_values('race')['id'].values

armed_vals = armed_data.groupby('race').count()['id'].reset_index().sort_values('race')['id'].values

chi_table = np.array([unarmed_vals, armed_vals])

chi_test = chi2_contingency(chi_table)

print('Chi Squared Statistic:', chi_test[0])

print('Degrees of Freedom:   ', chi_test[2])

print('P Value:              ', chi_test[1])

print('SIGNIFICANCE' if chi_test[1] < 0.05 else 'NO SIGNIFICANCE')
unarmed_ages = unarmed_data.groupby('age').count()['id'].reset_index().sort_values('age')

armed_ages = armed_data.groupby('age').count()['id'].reset_index().sort_values('age')



unarmed_vals = []

armed_vals = []

for age in range(100):

    zeros = 0

    try:

        unarmed_vals.append(unarmed_ages[unarmed_ages['age'] == age]['id'].reset_index(drop=True)[0])

    except:

        unarmed_vals.append(0)

        zeros += 1

    try:

        armed_vals.append(armed_ages[armed_ages['age'] == age]['id'].reset_index(drop=True)[0])

    except:

        armed_vals.append(0)

        zeros += 1

    if zeros > 1:

        unarmed_vals.pop()

        armed_vals.pop()

chi_table = np.array([unarmed_vals, armed_vals])

chi_test = chi2_contingency(chi_table)

print('Chi Squared Statistic:', chi_test[0])

print('Degrees of Freedom:   ', chi_test[2])

print('P Value:              ', chi_test[1])

print('SIGNIFICANCE' if chi_test[1] < 0.05 else 'NO SIGNIFICANCE')
from wordcloud import WordCloud

import requests

from PIL import Image

from io import BytesIO



# Obtain image mask

response = requests.get('https://www.kindpng.com/picc/m/672-6720557_us-outline-usa-map-vector-png-transparent-png.png')

usa_mask = np.array(Image.open(BytesIO(response.content)))[:,:,2]

usa_mask = 255 * (usa_mask > 50)



# Get a dict of unarmed names

unarmed_names = data[data['armed'] == 'unarmed']['name'].values

unarmed_names_dict = dict()

for name in unarmed_names:

    unarmed_names_dict[name] = 1



# Create wordcloud

wc = WordCloud(background_color='white', mask=usa_mask, max_words=1000,

               contour_width=10, max_font_size=20, colormap='plasma').generate_from_frequencies(unarmed_names_dict)



# Display wordcloud

plt.figure(figsize=(10, 10))

plt.imshow(wc, interpolation='bilinear')

plt.axis('off')

plt.show()
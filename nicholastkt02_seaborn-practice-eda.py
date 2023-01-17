import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

import seaborn as sns

import plotly.graph_objects as go

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv('../input/police-deadly-force-usage-us/fatal-police-shootings-data.csv')
data.head(10)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize = (12, 8))



sns.countplot(x = "race", orient = "v", ax = ax1, data = data)

sns.countplot(x = "gender", orient = "v", ax = ax2, data = data)

sns.countplot(x = "signs_of_mental_illness", orient = "v", ax = ax3, data = data)

sns.countplot(x = "age", orient = "v", ax = ax4, data = data)

ax4.set_xticks(range(0, 100, 10)) #changing the range for x axis, 100 is the max lim, 0 is min limit, and by 10

ax4.set_xticklabels(range(0, 100, 10)) # we have to manually change the limits and jumps as well

plt.tight_layout()

plt.show()
# Loading data

us_census_data = pd.read_csv("../input/us-census-demographic-data/acs2015_county_data.csv")



us_census_data.head(6)

# Getting the population proportions

total_population = us_census_data['TotalPop'].sum()

print(total_population)





race_proportions = pd.DataFrame(['White', 'Hispanic', 'Black', 'Asian', 'Native'], columns = ['Race'])



race_proportions['Population'] = race_proportions['Race'].apply(

    lambda y:us_census_data.apply(lambda x: x['TotalPop'] * x[y] / total_population, axis = 1).sum())



race_proportions['Killed In Police Shootings'] = race_proportions['Race'].apply(

    lambda x:data[data['race'] == x[0]].shape[0] * 100 / data.shape[0])



print(race_proportions)
# Plotting the proportions

race_proportions = race_proportions.melt(id_vars = "Race") 

#melt unpivots data frame from wide to long format, id_vars: column(s) to use as identifier variables

print(race_proportions)
# Plotting our data

fig, ax = plt.subplots(1, 1, figsize = (10, 6)) # 1 row and 1 column

sns.barplot(x = 'value', 

            y = 'Race', 

            data = race_proportions, 

            hue = "variable", #seperates the bars into the blue & orange

            ax = ax,

            palette = ['#0390fc', '#ff3321'])



fig, ax = plt.subplots(1, 1, figsize = (10, 6)) # 1 row and 1 column

sns.barplot(x = 'value', 

            y = 'Race', 

            data = race_proportions, 

            hue = "variable", #seperates the bars into the blue & orange

            ax = ax,

            palette = ['#0390fc', '#ff3321'])



# Annotations

for p in ax.patches:

    width = p.get_width()

    plt.text(8 + p.get_width(), #the distance the 5 figures are from the end of the bar, i.e. dist between 61.63% and the bar

             p.get_y() + 0.4 * p.get_height(), # the placing of the % alignment with the bars, try tuning it

             '{:.2f}%'.format(width), #{:.2}: format float to 2 decimal places, i.e 3.14151296 -> 3.14

            ha = 'center', va = 'center')

    

    

# Customise

ax.set_title('% of deaths from police shootings/ncompared to percentage of population by Race', fontsize = 16)

ax.tick_params(axis = 'both', labelsize = 12)

for spine in ax.spines.values():

    spine.set_visible(False)

ax.set_xlabel('')

ax.set_ylabel('')

ax.set_xticks([])

plt.legend(frameon=False, fontsize=12, ncol=2)

plt.tight_layout()

plt.show()
data.head()
state_counts = data.groupby(by = "state").agg({'id': 'count'}).reset_index()

state_counts



#dataframe.agg = aggregate using one or more operations over the specified axis.

#details: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.agg.html
fig = go.Figure(data = go.Choropleth(

    locations = state_counts['state'],

    z = state_counts['id'],

    locationmode = "USA-states",

    colorscale = "Viridis",

    colorbar_title = "Deaths"))



fig.update_layout(

    title_text = "Police Shooting Deaths by US States",

    geo_scope = "usa")



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



state_pops['State Codes'] = state_pops['State'].apply(lambda x:state_codes[x])

state_counts['Pop'] = state_counts['state'].apply(

    lambda x:state_pops[state_pops['State Codes'] == x].reset_index()['2018 Population'][0])
state_counts.head(20)
fig = go.Figure(data = go.Choropleth(

    locations = state_counts['state'],

    z = state_counts['id'] / state_counts['Pop'] * 100000,

    locationmode = 'USA-states',

    colorscale = 'Viridis',

    colorbar_title = "Deaths per 100,000"))



fig.update_layout(

    title_text = 'Police Shooting Deaths by US States per 100,000 People',

    geo_scope = "usa")



fig.show()
from datetime import date



data['date'] = pd.to_datetime(data['date'])

newd = data.groupby(pd.Grouper(key = 'date', # Groupby key

                               freq = 'M')).count().reset_index()[['date',  'id']]

# freq: This will groupby the specified frequency if the target selection is a datetime-like object

newd.head()

    
newd['date_ordinal'] = newd['date'].apply(lambda x:x.toordinal())

# The reason the date is changed to gregorian calender is that later when we plot the regplot, seaborn does not recognize 

# the original date format, but does recognise the gregorian format.

newd.head()
fig, ax = plt.subplots(1, 1, figsize = (12,4))

sns.regplot(x = 'date_ordinal', y = 'id', ci = 95, ax = ax, data = newd)
fig, ax = plt.subplots(1, 1, figsize = (12,4))

sns.regplot(x = 'date_ordinal', y = 'id', ci = 95, ax = ax, data = newd)



year_labels = [newd['date_ordinal'].min() + (x * 365) for x in range(6)]

ax.set_xticks(year_labels)



plt.show()
fig, ax = plt.subplots(1, 1, figsize = (12,4))

sns.regplot(x = 'date_ordinal', y = 'id', ci = 95, ax = ax, data = newd)



year_labels = [newd['date_ordinal'].min() + (x * 365) for x in range(6)]

ax.set_xticks(year_labels)

ax.set_xticklabels([2015, 2016, 2017, 2018, 2019, 2020])

ax.set_xlabel('Year')

ax.set_ylabel('Deaths')



plt.title('US Police Shooting Deaths over Time', fontsize = 14)

plt.show()
unarmed = ['unarmed', 'toy weapon', np.nan, 'undetermined', 'flashlight'] #creating a list of what exactly is unarmed to be applied later

           

data['is_armed'] = data['armed'].apply(lambda x:'Armed' if x not in unarmed else 'Unarmed')

unarmed_data = data[data['is_armed'] == 'Unarmed']

armed_data = data[data['is_armed'] == 'Armed']
armed_data.head()
unarmed_data.head()
fig = plt.figure(figsize = (12, 12))

gs = GridSpec(4, 2)

ax0 = fig.add_subplot(gs[0, :])

ax1 = fig.add_subplot(gs[1, 0])

ax2 = fig.add_subplot(gs[1, 1])

ax3 = fig.add_subplot(gs[2, 0])

ax4 = fig.add_subplot(gs[2, 1])

ax5 = fig.add_subplot(gs[3, 0])

ax6 = fig.add_subplot(gs[3, 1])





fig.suptitle('Unarmed vs Armed', y = 1.03, fontsize = 18)



sns.countplot('is_armed', ax = ax0, data = data, order = ['Unarmed', 'Armed'])

ax0.set_xlabel('')



# Race

sns.barplot(x = 'race', y = 'race', orient = 'v', ax = ax1, data = unarmed_data, 

           estimator = lambda x:len(x) / len(unarmed_data) * 100,

           order = ['W', 'B', 'H', 'A', 'N', 'O'])

sns.barplot(x = 'race', y = 'race', orient = 'v', ax = ax2, data = armed_data,

           estimator = lambda x:len(x) / len(armed_data) * 100,

           order = ['W', "B", "H", "A", 'N', 'O'])

for ax in [ax1, ax2]:

    ax.set_ylabel('percent')

    

# Gender

sns.barplot(x = 'gender', y = 'gender', orient = 'v', ax = ax3, data = unarmed_data, 

           estimator = lambda x:len(x) / len(unarmed_data) * 100,

           order = ['M', 'F'])

sns.barplot(x = 'gender', y = 'gender', orient = 'v', ax = ax4, data = armed_data, 

           estimator = lambda x: len(x) / len(armed_data) * 100,

           order = ['M', 'F'])



# Age

sns.barplot(x = 'age', y = 'age', data = unarmed_data, ax = ax5, 

           estimator = lambda x:len(x) / len(unarmed_data) * 100)

sns.barplot(x = 'age', y = 'age', data = armed_data, ax = ax6,

           estimator = lambda x:len(x) / len(armed_data) * 100)



for ax in [ax5, ax6]:

    ax.set_xticks(range(0, 90, 10))

    ax.set_xticklabels(range(0, 90, 10))

    ax.set_ylabel('percent')



plt.tight_layout()

plt.show()
from wordcloud import WordCloud

import requests

from PIL import Image

from io import BytesIO



# Obtain image mask

response = '../input/state-outline-png/672-6720557_us-outline-usa-map-vector-png-transparent-png.png'

usa_mask = np.array(Image.open(response))[:,:,2]

usa_mask = 255 * (usa_mask > 50)



# Get a dict of unarmed names

unarmed_names = data[data['armed'] == 'unarmed']['name'].values

unarmed_names_dict = dict()

for name in unarmed_names:

    unarmed_names_dict[name] = 1



# Create wordcloud

wc = WordCloud(background_color='white', mask=usa_mask, max_words=1000, contour_width=10, max_font_size=20, colormap='plasma').generate_from_frequencies(unarmed_names_dict)



# Display wordcloud

plt.figure(figsize=(10, 10))

plt.imshow(wc, interpolation='bilinear')

plt.axis('off')

plt.show()

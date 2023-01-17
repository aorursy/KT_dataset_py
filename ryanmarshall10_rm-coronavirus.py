import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Load in the data:



recovered = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

deaths = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

confirmed = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

open_line_list = pd.read_csv('../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv')

line_list = pd.read_csv('../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv')

data = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
#Take a look at what it looks like:

confirmed.head()
#make a df that just has the latest numbers



recovered_latest = recovered[recovered['3/14/20']!=0].groupby('Country/Region').sum()

recovered_latest = recovered_latest.iloc[:,-1].sort_values(ascending = False)

recovered_latest = pd.DataFrame(recovered_latest)

recovered_latest.reset_index(inplace = True)

recovered_latest.head()
confirmed_latest = confirmed[confirmed['3/14/20']!=0].groupby('Country/Region').sum()

confirmed_latest = confirmed_latest.iloc[:,-1].sort_values(ascending = False)

confirmed_latest = pd.DataFrame(confirmed_latest)

confirmed_latest.reset_index(inplace = True)

confirmed_latest.head()
deaths_latest = deaths[deaths['3/14/20']!=0].groupby('Country/Region').sum()

deaths_latest = deaths_latest.iloc[:,-1].sort_values(ascending = False)

deaths_latest = pd.DataFrame(deaths_latest)

deaths_latest.reset_index(inplace = True)

deaths_latest.head()
#Convert countries to country codes for plotting chloropleth:



from iso3166 import countries



def code_map(x):

    if x == 'Iran':

        x = 'Iran, Islamic Republic of'

    elif x == 'Korea, South':

        x = 'Korea, Republic of'

    elif x == 'United Kingdom':

        x = 'United Kingdom of Great Britain and Northern Ireland'

    elif x == 'Cruise Ship':

        return x

    elif x == 'Taiwan*':

        x = 'Taiwan, Province of China'

    elif x == 'Vietnam':

        x = 'Viet Nam'

    elif x == 'Russia':

        x = 'Russian Federation'

    elif x == 'Brunei':

        x = 'Brunei Darussalam'

    elif x == 'Moldova':

        x ='Moldova, Republic of'

    elif x == 'Bolivia':

        x = 'Bolivia, Plurinational State of'

    elif x == 'Reunion':

        x = 'Réunion'

    elif x == 'Congo (Kinshasa)':

        x = 'Congo'

    elif x == 'Venezuela':

        x = 'Venezuela, Bolivarian Republic of'

    elif x == 'Curacao':

        x = 'Curaçao'

    elif x == "Cote d'Ivoire":

        x = "Côte d'Ivoire"

    return countries.get(x).alpha3



deaths_latest['Code'] = deaths_latest['Country/Region'].apply(code_map)

recovered_latest['Code'] = recovered_latest['Country/Region'].apply(code_map)

confirmed_latest['Code'] = confirmed_latest['Country/Region'].apply(code_map)
import plotly.graph_objects as go
#Chloropleth plots:



data = dict(

        type = 'choropleth',

        locations = deaths_latest['Code'],

        colorscale = 'Jet',

        z = np.log10(deaths_latest['3/14/20']),

        text = deaths_latest['Country/Region'],

        colorbar=dict(title='Deaths', tickprefix='10^'),

      ) 

#colorbar=dict(title='Count (Log)', tickprefix='1.e')

#colorbar = {'title' : 'COVID-19 deaths'}

layout = dict(

    title = 'COVID-19 deaths',

    geo = dict(

        showframe = False,

        projection = {'type':'equirectangular'}

    )

)



choromap = go.Figure(data = [data],layout = layout)

#iplot(choromap)

choromap.show()
data2 = dict(

        type = 'choropleth',

        locations = confirmed_latest['Code'],

        colorscale = 'Jet',

        z = np.log10(confirmed_latest['3/14/20']),

        text = confirmed_latest['Country/Region'],

        colorbar=dict(title='Cases', tickprefix='10^'),

      ) 

#colorbar=dict(title='Count (Log)', tickprefix='1.e')

#colorbar = {'title' : 'COVID-19 deaths'}

layout2 = dict(

    title = 'COVID-19 cases',

    geo = dict(

        showframe = False,

        projection = {'type':'equirectangular'}

    )

)



choromap2 = go.Figure(data = [data2],layout = layout2)

#iplot(choromap)

choromap2.show()
data3 = dict(

        type = 'choropleth',

        locations = recovered_latest['Code'],

        colorscale = 'Jet',

        z = np.log10(recovered_latest['3/14/20']),

        text = recovered_latest['Country/Region'],

        colorbar=dict(title='Recoveries', tickprefix='10^'),

      ) 

#colorbar=dict(title='Count (Log)', tickprefix='1.e')

#colorbar = {'title' : 'COVID-19 deaths'}

layout3 = dict(

    title = 'COVID-19 recoveries',

    geo = dict(

        showframe = False,

        projection = {'type':'equirectangular'}

    )

)



choromap3 = go.Figure(data = [data3],layout = layout3)

#iplot(choromap)

choromap3.show()
#Look at just US cases:

confirmed_US = confirmed[confirmed['Country/Region']=='US']

confirmed_US = confirmed_US[confirmed_US['3/14/20']!= 0]

confirmed_US_latest = confirmed_US.iloc[:,[0,-1]]
states = {

        'AK': 'Alaska',

        'AL': 'Alabama',

        'AR': 'Arkansas',

        'AS': 'American Samoa',

        'AZ': 'Arizona',

        'CA': 'California',

        'CO': 'Colorado',

        'CT': 'Connecticut',

        'DC': 'District of Columbia',

        'DE': 'Delaware',

        'FL': 'Florida',

        'GA': 'Georgia',

        'GU': 'Guam',

        'HI': 'Hawaii',

        'IA': 'Iowa',

        'ID': 'Idaho',

        'IL': 'Illinois',

        'IN': 'Indiana',

        'KS': 'Kansas',

        'KY': 'Kentucky',

        'LA': 'Louisiana',

        'MA': 'Massachusetts',

        'MD': 'Maryland',

        'ME': 'Maine',

        'MI': 'Michigan',

        'MN': 'Minnesota',

        'MO': 'Missouri',

        'MP': 'Northern Mariana Islands',

        'MS': 'Mississippi',

        'MT': 'Montana',

        'NA': 'National',

        'NC': 'North Carolina',

        'ND': 'North Dakota',

        'NE': 'Nebraska',

        'NH': 'New Hampshire',

        'NJ': 'New Jersey',

        'NM': 'New Mexico',

        'NV': 'Nevada',

        'NY': 'New York',

        'OH': 'Ohio',

        'OK': 'Oklahoma',

        'OR': 'Oregon',

        'PA': 'Pennsylvania',

        'PR': 'Puerto Rico',

        'RI': 'Rhode Island',

        'SC': 'South Carolina',

        'SD': 'South Dakota',

        'TN': 'Tennessee',

        'TX': 'Texas',

        'UT': 'Utah',

        'VA': 'Virginia',

        'VI': 'Virgin Islands, U.S.',

        'VT': 'Vermont',

        'WA': 'Washington',

        'WI': 'Wisconsin',

        'WV': 'West Virginia',

        'WY': 'Wyoming',

        'DP': 'Diamond Princess',

        'GP': 'Grand Princess'

}



states2 = {}

for key in states:

    val = states[key]

    states2[val] = key
confirmed_US_latest['Code'] = confirmed_US_latest['Province/State'].copy().apply(lambda x: states2[x])
data4 = dict(type='choropleth',

            locations = confirmed_US_latest['Code'],

            z = confirmed_US_latest['3/14/20'],

            locationmode = 'USA-states',

            text = confirmed_US_latest['Province/State'],

            colorbar = {'title':"Cases"}

            ) 



layout4 = dict(title = 'COVID-19 cases',

              geo = dict(scope='usa',

                         showlakes = True,

                         lakecolor = 'rgb(85,173,240)')

             )



choromap4 = go.Figure(data = [data4],layout = layout4)

choromap4.show()
deaths_latest.rename(columns = {'3/14/20':'Deaths'}, inplace = True)

confirmed_latest.rename(columns = {'3/14/20':'Cases'}, inplace = True)

recovered_latest.rename(columns = {'3/14/20':'Recoveries'}, inplace = True)
latest = pd.merge(deaths_latest, confirmed_latest, how = 'outer', on = ['Country/Region','Code'])

latest = pd.merge(latest, recovered_latest, how = 'outer', on = ['Country/Region','Code'] )

#latest.drop(columns = 'index', inplace = True)
latest = latest[['Country/Region','Code','Cases','Deaths','Recoveries']]

latest.fillna(value = 0, inplace = True)

latest.head()
sns.set_context("poster")

fig = plt.figure(figsize = (20,10))

sns.barplot(x = latest.sort_values(by = 'Cases', ascending = False)['Code'][:10],

            y = latest.sort_values(by = 'Cases', ascending = False)['Cases'][:10])



plt.tight_layout()

                         
#Create df with just ten highest countries by confirmed cases:



topten = confirmed.groupby('Country/Region').sum().sort_values(by = '3/14/20', ascending = False)[:10].transpose()

top_rec = recovered.groupby('Country/Region').sum().sort_values(by = '3/14/20', ascending = False).transpose()

top_death = deaths.groupby('Country/Region').sum().sort_values(by = '3/14/20', ascending = False).transpose()
topten.drop(['Lat', 'Long'], axis = 0, inplace = True)

top_rec.drop(['Lat', 'Long'], axis = 0, inplace = True)

top_death.drop(['Lat', 'Long'], axis = 0, inplace = True)
topten.rename(columns = {'Korea, South': 'South Korea'}, inplace = True)

top_rec.rename(columns = {'Korea, South': 'South Korea'}, inplace = True)

top_death.rename(columns = {'Korea, South': 'South Korea'}, inplace = True)

new_cases = topten.diff()

topten.reset_index(inplace = True)

top_rec.reset_index(inplace = True)

top_death.reset_index(inplace = True)

sns.set(style="whitegrid")

sns.set_context("talk")

# Initialize the matplotlib figure

f, ax = plt.subplots(3,1, figsize=(20, 20))





sns.set_color_codes("pastel")

sns.barplot(x="index", y="China", data=topten, ax = ax[0],

            label="Confirmed", color="b")





sns.set_color_codes("muted")

sns.barplot(x="index", y="China", data=top_rec, ax = ax[0],

            label="Recovered", color="b")



sns.set_color_codes("pastel")

sns.barplot(x="index", y="Italy", data=topten, ax = ax[1],

            label="Confirmed", color="b")



sns.set_color_codes("muted")

sns.barplot(x="index", y="Italy", data=top_rec, ax = ax[1],

            label="Recovered", color="b")



sns.set_color_codes("pastel")

sns.barplot(x="index", y="US", data=topten, ax = ax[2],

            label="Confirmed", color="b")



sns.set_color_codes("muted")

sns.barplot(x="index", y="US", data=top_rec, ax = ax[2],

            label="Recovered", color="b")



ax[0].legend(ncol=1, loc="upper left", frameon=True)

sns.despine(left=True, bottom=True)

ax[0].set(ylabel="Cases",xlabel="Day", title = "China")

ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, horizontalalignment='right')



ax[1].legend(ncol=1, loc="upper left", frameon=True)

sns.despine(left=True, bottom=True)

ax[1].set(ylabel="Cases",xlabel="Day", title = "Italy")

ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, horizontalalignment='right')



ax[2].legend(ncol=1, loc="upper left", frameon=True)

sns.despine(left=True, bottom=True)

ax[2].set(ylabel="Cases",xlabel="Day", title = "USA")

ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=45, horizontalalignment='right')



plt.tight_layout()

f.show()
sns.set(style="whitegrid")

sns.set_context("talk")

# Initialize the matplotlib figure

f, ax = plt.subplots(3,1, figsize=(20, 20))





sns.set_color_codes("pastel")

sns.barplot(x="index", y="China", data=topten, ax = ax[0],

            label="Confirmed", color="b")





sns.set_color_codes("muted")

sns.barplot(x="index", y="China", data=top_death, ax = ax[0],

            label="Deaths", color="b")



sns.set_color_codes("pastel")

sns.barplot(x="index", y="Italy", data=topten, ax = ax[1],

            label="Confirmed", color="b")



sns.set_color_codes("muted")

sns.barplot(x="index", y="Italy", data=top_death, ax = ax[1],

            label="Deaths", color="b")



sns.set_color_codes("pastel")

sns.barplot(x="index", y="US", data=topten, ax = ax[2],

            label="Confirmed", color="b")



sns.set_color_codes("muted")

sns.barplot(x="index", y="US", data=top_death, ax = ax[2],

            label="Deaths", color="b")



ax[0].legend(ncol=1, loc="upper left", frameon=True)

sns.despine(left=True, bottom=True)

ax[0].set(ylabel="Cases",xlabel="Day", title = "China")

ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, horizontalalignment='right')



ax[1].legend(ncol=1, loc="upper left", frameon=True)

sns.despine(left=True, bottom=True)

ax[1].set(ylabel="Cases",xlabel="Day", title = "Italy")

ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, horizontalalignment='right')



ax[2].legend(ncol=1, loc="upper left", frameon=True)

sns.despine(left=True, bottom=True)

ax[2].set(ylabel="Cases",xlabel="Day", title = "USA")

ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=45, horizontalalignment='right')



plt.tight_layout()

f.show()
#Compare different countries when their outbreaks first started: adjust for lag



lag_df = topten[['index','China','Italy','South Korea','Iran','US']].copy()

lag_df['China'] = lag_df['China'].shift(8)

lag_df['Italy'] = lag_df['Italy'].shift(-29)

lag_df['South Korea'] = lag_df['South Korea'].shift(-26)

lag_df['US'] = lag_df['US'].shift(-40)

lag_df['Iran'] = lag_df['Iran'].shift(-31)
melted = pd.melt(lag_df[:20], id_vars="index", var_name="country", value_name="Confirmed")
g, axr = plt.subplots(1,1, figsize=(20, 10))

sns.barplot(x='index', y='Confirmed', hue='country', data=melted, ax = axr)

axr.set(ylabel="Cases",xlabel="Day (Adjusted for lag)", title = "COVID-19 Confirmed Cases")

axr.set_xticklabels(axr.get_xticklabels(), rotation=45, horizontalalignment='right')

g.show()
#The USA is on track with China to head for 100,000s of cases, maybe more with the lack of testing.

#South Korea, which has testing, is already flattening the curve.
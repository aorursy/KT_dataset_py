# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# using pandas read_csv

#dataset = '/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv'

dataset = '/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv'



df = pd.read_csv(dataset)
df.head()
df.tail()
EMPTY_VAL = "Unknown"



def fillState(state, country):

    if state == EMPTY_VAL: return country

    return state
df_T = df.copy()

df_T.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country'}, inplace=True)

df_T['Province/State'].fillna(EMPTY_VAL, inplace=True)

df_T['Date'] = pd.to_datetime(df_T['Date'], infer_datetime_format=True)
df_T.loc[: , ['Date', 'Country', 'Province/State', 'Confirmed', 'Deaths', 'Recovered']].groupby(['Country', 'Province/State']).max().groupby(['Date', 'Country']).sum().sort_values(by='Confirmed', ascending=False).reset_index()[:15].style.background_gradient(cmap='rainbow')
#df_T.loc[: , ['Date', 'Country', 'Province/State', 'Confirmed', 'Deaths', 'Recovered']].groupby(['Country', 'Province/State']).max().sort_values(by='Confirmed', ascending=False).reset_index()[:15].style.background_gradient(cmap='rainbow')
import plotly.express as px

top_10_countries = df_T.loc[: , ['Date', 'Country', 'Province/State', 'Confirmed', 'Deaths', 'Recovered']].groupby(['Country', 'Province/State']).max().groupby(['Date', 'Country']).sum().sort_values(by='Confirmed', ascending=False).reset_index().loc[:, 'Country'][:10]

df_plot = df_T.loc[df_T.Country.isin(top_10_countries), ['Date', 'Country', 'Province/State', 'Confirmed', 'Deaths', 'Recovered']].groupby(['Date', 'Country', 'Province/State']).max().groupby(['Date', 'Country']).sum().reset_index()



fig = px.line(df_plot, x="Date", y="Confirmed", color='Country')

fig.update_layout(title='No.of Confirmed Cases per Day for Top 10 Countries',

                   xaxis_title='Date',

                   yaxis_title='No.of Confirmed Cases')

fig.show()
fig = px.line(df_plot, x="Date", y="Deaths", color='Country')

fig.update_layout(title='No.of Death Cases per Day for Top 10 Countries',

                   xaxis_title='Date',

                   yaxis_title='No.of Death Cases')

fig.show()
df.iloc[np.r_[0:5, -6:-1], :]
df
df.info()
df.describe()
df.shape
df.isnull().sum()
df_T.loc[: , ['Date', 'Country', 'Province/State', 'Confirmed', 'Deaths', 'Recovered']].groupby(['Country', 'Province/State']).max().groupby(['Date', 'Country']).sum().reset_index().select_dtypes(include='float64').sum()
# Let's get rid of the Sno column as it's redundant

df.drop(['SNo'], axis=1, inplace=True)
df.drop('Last Update', axis=1, inplace=True)
df.tail()
df['Province/State'].value_counts()
tempState = df['Province/State'].mode()

#print(tempState)

#df['Province/State'].fillna(tempState, inplace=True)
df.tail()
df.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country'}, inplace=True)
from datetime import datetime

# 1/22/2020 12:00

# 1/26/2020 23:00

# 1/23/20 12:00 PM

# 2020-01-02 23:33:00

# 

def try_parsing_date_time(text):

    for fmt in ('%Y-%m-%d', '%d.%m.%Y', '%m/%d/%Y', '%m/%d/%Y %h:%m', '%m/%d/%Y %H:%M','%m/%d/%Y %H:%M:%S','%m/%d/%y %I:%M %p', '%m/%d/%Y %I:%M %p', '%Y-%d-%m %H:%M:%S'):

        try:

            return datetime.strptime(text, fmt)

        except ValueError:

            pass

    raise ValueError('no valid date time format found', text)





def try_parsing_date(text):

    for fmt in ('%m/%d/%Y', '%m/%d/%y', '%Y-%d-%m', '%d.%m.%Y'):

        try:

            return datetime.strptime(text, fmt)

        except ValueError:

            pass

    raise ValueError('no valid date format found', text)
'''

dateTime = df['Last Update'].apply(try_parsing_date)

df['DateTime'] = dateTime

df['Time'] = dateTime.dt.time

df['Date'] = dateTime.dt.date

df.drop('Last Update', axis=1, inplace=True)

'''
df['Date']
df['Date'] = df['Date'].apply(try_parsing_date_time)
df['Date']
df[df['Province/State'] == 'Taiwan']['Country'] = 'Taiwan'

df[df['Province/State'] == 'Hong Kong']['Country'] = 'Hong Kong'
df.replace({'Country': 'Mainland China'}, 'China', inplace=True)
#df[['Confirmed', 'Deaths', 'Recovered']].astype('int32')
df.info()
#df['Date'].unique()
df.Country.unique()
df[['Country', 'Confirmed']].groupby(['Country']).count()
df['Province/State'].unique()
df_IN = df.loc[(df.Country == 'India') & (df.Date >= '2020-06-10'), :]

df_IN.loc[:, '%Change'] = df_IN['Confirmed'].transform(lambda x : round(100 * (x - x.shift(1)) / x.shift(1)))

df_IN
import plotly.express as px

%matplotlib inline



df_plot = df_IN.loc[:,['Date', 'Confirmed', '%Change']]

df_plot.loc[:, 'Date'] = df_plot.Date.dt.strftime("%m%d")



px.bar(df_plot, x='Date', y='Confirmed')
px.bar(df_plot, x='Date', y='%Change')
df['Country'].value_counts()
df.Country.nunique()
import plotly.express as px

pxdf = px.data.gapminder()



country_isoAlpha = pxdf[['country', 'iso_alpha']].drop_duplicates()

country_isoAlpha.rename(columns = {'country':'Country'}, inplace=True)

country_isoAlpha.set_index('Country', inplace=True)

country_map = country_isoAlpha.to_dict('index')
def getCountryIsoAlpha(country):

    try:

        return country_map[country]['iso_alpha']

    except:

        return country
df['iso_alpha'] = df['Country'].apply(getCountryIsoAlpha)

df.info()
df_plot = df.loc[:,['Date', 'Country', 'Confirmed']]

df_plot.loc[:, 'Date'] = df_plot.Date.dt.strftime("%Y-%m-%d")

df_plot.loc[:, 'Size'] = np.where(df_plot['Country']=='China', df_plot['Confirmed'], df_plot['Confirmed']*200)

fig = px.scatter_geo(df_plot.groupby(['Date', 'Country']).max().reset_index(),

                     locations="Country",

                     locationmode = "country names",

                     hover_name="Country",

                     color="Confirmed",

                     animation_frame="Date", 

                     size='Size',

                     #projection="natural earth",

                     title="Rise of Coronavirus Confirmed Cases")

fig.show()

df[['Date', 'iso_alpha', 'Confirmed']].groupby('iso_alpha').max()
df_plot = df.groupby('iso_alpha').max().reset_index()

fig = px.choropleth(df_plot, locations="iso_alpha",

                    color="Confirmed", 

                    hover_name="iso_alpha", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Plasma)

fig.show()
df_plot = df.groupby('iso_alpha').max().reset_index()

fig = px.choropleth(df_plot, locations="iso_alpha",

                    color="Deaths", 

                    hover_name="iso_alpha", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Plasma)

fig.show()
df_plot = df.groupby('iso_alpha').max().reset_index()

fig = px.choropleth(df_plot, locations="iso_alpha",

                    color="Recovered", 

                    hover_name="iso_alpha")

fig.show()
df.groupby('Country')['Confirmed'].max()
df.groupby('Country')['Confirmed'].max().sort_values(ascending=False)[0:10]
df.groupby('Country')['Deaths'].max().sort_values(ascending=False)
df.groupby('Country')['Recovered'].max()
# df[df.Country == 'China'][['Province/State', 'Confirmed']].groupby('Province/State').max()
df['Province/State'].unique()

import plotly.graph_objs as go

fig = go.Figure(go.Choroplethmapbox(geojson='china', locations=df['Province/State'].unique(), z=df[df.Country == 'China'][['Province/State', 'Confirmed']].groupby('Province/State').max(),

                                    colorscale='Cividis', zmin=0, zmax=17,

                                    marker_opacity=0.5, marker_line_width=0))

fig.update_layout(mapbox_style="carto-positron",

                  mapbox_zoom=3, mapbox_center = {"lat": 35.8617, "lon": 104.1954})
df.sort_values(by='Date')['Date'][0]
df['Date'].max()
df.groupby('Date')[['Confirmed', 'Deaths', 'Recovered']].max().reset_index()
df[df.Country == 'India'][['Date', 'Country', 'Confirmed']].groupby('Country').plot(x='Date', y='Confirmed', kind='line', legend='Confirmed cases in India')
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



plt.rcParams["figure.figsize"] = (16,9)

plt.figure(figsize=(16,9));
df[['Confirmed', 'Deaths', 'Recovered']].max().plot(kind='bar')
plt.figure(figsize=(12,7))

chart = sns.countplot(data=df, x='Country', palette='Set1')

chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right', fontweight='light');
chart = sns.catplot(

    data=df,

    x='Country',

    kind='count',

    palette='Set1',

    row='Date',

    aspect=3,

    height=3

)
df.loc[(df['Country'] == "China") & (df['Date'] == '2020-01-23 12:00:00')]
df.loc[df.Country == "China"]
import ipywidgets as widgets

from ipywidgets import interact, interact_manual



from IPython.display import display
ALL = 'ALL'

def uniqueSortedValues(fromList):

    fromList.dropna(inplace=True)

    values = fromList.unique().tolist()

    values.sort()

    values.insert(0, ALL)

    return values



def showRecordsPerDay():

    df_plot = df.loc[(df['Country'] == country) & (df['Province/State'] == state)].groupby(['Country']).max()

   

    return None
#df['Province/State'].unique()
dropdown_country = widgets.Dropdown(options = uniqueSortedValues(df['Country']), description = 'Country')
dropdown_state = widgets.Dropdown(options = uniqueSortedValues(df['Province/State']), description = 'State')
output = widgets.Output()
def common_filtering_output(country, state):

    output.clear_output()

    if country == 'ALL' and state == 'ALL':

        out = df

    elif country == 'ALL':

        out = df[df['Province/State'] == state]

    elif state == 'ALL':

        out = df[df.Country == country]

    else:

        out = df[(df.Country == country) & (df['Province/State'] == state)]

    

    with output:

        display(out)

    
def dropdown_country_eventhandler(change):

    common_filtering_output(change.new, dropdown_state.value)
def dropdown_state_eventhandler(change):

    common_filtering_output(dropdown_country.value, change.new)
dropdown_country.observe(dropdown_country_eventhandler, names='value')

dropdown_state.observe(dropdown_state_eventhandler, names='value')
#display(dropdown_country)

#display(dropdown_state)
df.loc[df['Country'] == 'China']
df[df.Country == 'China'].tail(10)
'''

column = 'Country'

country = 'China'

start_date = '2020-02-09 23:20:00'

df_plot = df.loc[(df[column] == country) & (df['Date'].dt.strftime('%Y-%m-%d') == '2020-02-08')]

chart = sns.catplot(

                data=df_plot,

                x='Country',

                y='Confirmed',

                kind = 'bar',

                palette='viridis',

                row='Date',

                aspect=1,

                height=5)

'''
@interact

def showRecordsPerDay(column = ['Confirmed', 'Deaths', 'Recovered'], country = df.Country.unique(), start_date = widgets.DatePicker(value = pd.to_datetime('2020-01-22')) ):

    """

    This function does display the records entered per day for each selected Country

    """

    try:

        df_plot = df.loc[(df['Country'] == country) & (df['Date'].dt.strftime('%Y-%m-%d') == start_date.strftime("%Y-%m-%d"))]

        chart = sns.catplot(

                data=df_plot,

                x='Country',

                y=column,

                kind = 'bar',

                palette='viridis',

                row='Date',

                aspect=1,

                height=3)

    except:

        print("No records exists for", country, "on", start_date.strftime("%Y-%m-%d"))

        return None

    

    return None

byState = df.groupby(['Country', 'Province/State']).size().unstack()

#print(byState)
plt.figure(figsize=(10,10))

g = sns.heatmap(

    byState, 

    square=True,

    cbar_kws={'fraction' : 0.01},

    cmap='RdYlGn_r',

    linewidth=1

)



#g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')

#g.set_yticklabels(g.get_yticklabels(), rotation=0, horizontalalignment='right')
df[df.Country == 'China'][['Province/State', 'Deaths', 'Recovered']].groupby('Province/State').max().plot(kind='bar')
f, axes = plt.subplots(1,3, figsize=(30,20))

df[df.Country == 'China'][df['Province/State'] != 'Hubei'].groupby('Province/State')['Confirmed', 'Deaths', 'Recovered'].max().plot(kind='pie', subplots=True, ax=axes, legend=None);
df[df.Country != 'China'].groupby('Date').max().plot(kind='line')
df[df.Country == 'China'].groupby('Date').max().plot(kind='line')
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

plt.rcParams['figure.figsize'] = (13, 13)

wordcloud = WordCloud(stopwords=STOPWORDS,background_color = 'black', width = 1000,  height = 1000, max_words = 121).generate(' '.join(df['Country']))

plt.imshow(wordcloud)

plt.axis('off')

plt.title('Most Popular Countries affected with Coronavirus',fontsize = 30)

plt.show()
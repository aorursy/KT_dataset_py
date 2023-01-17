import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt 

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly_express as px

import cufflinks as cf

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)

from wordcloud import WordCloud,STOPWORDS



import warnings            

warnings.filterwarnings("ignore") 
space_data = pd.read_csv('/kaggle/input/all-space-missions-from-1957/Space_Corrected.csv')

space_data.head()
space_data = space_data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
space_data.columns = ['Company Name', 'Location', 'Datum', 'Detail', 'Status Rocket', 'Rocket', 'Status Mission']
space_data['DateTime'] = pd.to_datetime(space_data['Datum'])



# Extract the launch year

space_data['Year'] = space_data['DateTime'].apply(lambda datetime: datetime.year)



# Extract the country of launch

space_data["Country"] = space_data["Location"].apply(lambda location: location.split(", ")[-1])
# check missing data



total_of_all = space_data.isnull().sum().sort_values(ascending=False)

percent_of_all = (space_data.isnull().sum()/space_data.isnull().count()).sort_values(ascending=False)

missing_data_test = pd.concat([total_of_all, percent_of_all], axis=1, keys=['Total', 'Percent'])

missing_data_test.head()
# handle missing values



space_data['Rocket'] = space_data['Rocket'].fillna(0)

space_data.head()
# lauches per company bar chart



company_launch_analysis = pd.DataFrame(space_data['Company Name'].value_counts().sort_values(ascending=False))

company_launch_analysis = company_launch_analysis.rename(columns={'Company Name':'Count'})



trace = go.Bar(x = company_launch_analysis.index[:20],

              y = company_launch_analysis['Count'][:20],

              marker = dict(color='rgba(255,155,128,0.5)',

              line = dict(color='rgb(0,0,0)', width=1.5)))



layout = go.Layout(title="Top 25 company with their no. of lauches",

                  xaxis=dict(title='Company Name',zeroline= False,

                         gridcolor='rgb(183,183,183)',showline=True),

                  yaxis=dict(title='Lauch Counts',zeroline= False,

                            gridcolor='rgb(183,183,183)',showline=True),

                  font=dict(family='Courier New, monospace', size=12, color='rgb(0,0,0)')

)

data = [trace]

fig = go.Figure(data = data, layout = layout)

iplot(fig)
fig = px.pie(company_launch_analysis, values=company_launch_analysis['Count'], names=company_launch_analysis.index,

             title='Company and Their Lauches Ratio in The World',

            )

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.update_layout(

    template='plotly_white'

)

fig.show()
country_launch_analysis = pd.DataFrame(space_data['Country'].value_counts().sort_values(ascending=False))

country_launch_analysis = country_launch_analysis.rename(columns={'Country':'Count'})



trace = go.Bar(x = country_launch_analysis.index[:15],

              y = country_launch_analysis['Count'][:15],

              marker = dict(color='rgba(125, 215, 180, 0.5)',

              line = dict(color='rgb(0,0,0)', width=1.5)))



layout = go.Layout(title="Top 15 countries with their no. of lauches",

                  xaxis=dict(title='Country Name',zeroline= False,

                         gridcolor='rgb(183,183,183)',showline=True),

                  yaxis=dict(title='Lauch Counts',zeroline= False,

                            gridcolor='rgb(183,183,183)',showline=True),

                  font=dict(family='Courier New, monospace', size=12, color='rgb(0,0,0)')

)

data = [trace]

fig = go.Figure(data = data, layout = layout)

iplot(fig)
fig = px.pie(country_launch_analysis, values=country_launch_analysis['Count'], names=country_launch_analysis.index,

             title='Countries and Their Lauches Ratio in The World',

            )

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.update_layout(

    template='plotly_white'

)

fig.show()
location_launch_analysis = pd.DataFrame(space_data['Location'].value_counts().sort_values(ascending=False))

location_launch_analysis = location_launch_analysis.rename(columns={'Location':'Count'})



trace = go.Bar(x = location_launch_analysis.index[:25],

              y = location_launch_analysis['Count'][:25],

              marker = dict(color='rgba(50, 160, 80, 0.5)',

              line = dict(color='rgb(0,0,0)', width=1.5)))



layout = go.Layout(title="Top 25 locations with their no. of lauches",

                  xaxis=dict(title='Loaction Name',zeroline= False,

                         gridcolor='rgb(183,183,183)',showline=True),

                  yaxis=dict(title='Lauch Counts',zeroline= False,

                            gridcolor='rgb(183,183,183)',showline=True),

                  font=dict(family='Courier New, monospace', size=12, color='rgb(0,0,0)')

)

data = [trace]

fig = go.Figure(data = data, layout = layout)

fig.update_layout(

    autosize=False,

    width=1200,

    height=700,)

iplot(fig)
fig = px.pie(location_launch_analysis, values=location_launch_analysis['Count'], names=location_launch_analysis.index,

             title='Locations and Their Lauches Ratio in The World',

            )

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.update_layout(

    template='plotly_white',

    autosize=False,

    width=1200,

    height=700

)

fig.show()
# status analysis



status_rocket_analysis = pd.DataFrame(space_data['Status Rocket'].value_counts().sort_values(ascending=False))

status_rocket_analysis = status_rocket_analysis.rename(columns={'Status Rocket':'Count'})



trace = go.Bar(x = status_rocket_analysis.index,

              y = status_rocket_analysis['Count'],

              marker = dict(color='rgba(115, 155, 214, 0.5)',

              line = dict(color='rgb(0,0,0)', width=1.5)))



layout = go.Layout(title="Status analysis",

                  xaxis=dict(title='Status of rockets',zeroline= False,

                         gridcolor='rgb(183,183,183)',showline=True),

                  yaxis=dict(title='Counts of rockets',zeroline= False,

                            gridcolor='rgb(183,183,183)',showline=True),

                  font=dict(family='Courier New, monospace', size=12, color='rgb(0,0,0)')

)

data = [trace]

fig = go.Figure(data = data, layout = layout)

iplot(fig)
# mission status



mission_rocket_analysis = pd.DataFrame(space_data['Status Mission'].value_counts().sort_values(ascending=False))

mission_rocket_analysis = mission_rocket_analysis.rename(columns={'Status Mission':'Count'})



trace = go.Bar(x = mission_rocket_analysis.index,

              y = mission_rocket_analysis['Count'],

              marker = dict(color='rgba(150, 200, 100, 0.5)',

              line = dict(color='rgb(0,0,0)', width=1.5)))



layout = go.Layout(title="Mission analysis",

                  xaxis=dict(title='Status of Mission',zeroline= False,

                         gridcolor='rgb(183,183,183)',showline=True),

                  yaxis=dict(title='Counts',zeroline= False,

                            gridcolor='rgb(183,183,183)',showline=True),

                  font=dict(family='Courier New, monospace', size=12, color='rgb(0,0,0)')

)

data = [trace]

fig = go.Figure(data = data, layout = layout)

iplot(fig)
fig = px.pie(mission_rocket_analysis, values=mission_rocket_analysis['Count'], names=mission_rocket_analysis.index,

             title='Mission Success and Failure Ratio')

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.update_layout(

    template='plotly_white'

)

fig.show()
# launches per year



year_launch_analysis = pd.DataFrame(space_data['Year'].value_counts().sort_values(ascending=False))

year_launch_analysis = year_launch_analysis.rename(columns={'Year':'Count'})



trace = go.Bar(x = year_launch_analysis.index,

              y = year_launch_analysis['Count'],

              marker = dict(color='rgba(255,155,128,0.5)',

              line = dict(color='rgb(0,0,0)', width=1.5)))



layout = go.Layout(title="Years with no. of lauches",

                  xaxis=dict(title='Years',zeroline= False,

                         gridcolor='rgb(183,183,183)',showline=True),

                  yaxis=dict(title='Lauch Counts',zeroline= False,

                            gridcolor='rgb(183,183,183)',showline=True),

                  font=dict(family='Courier New, monospace', size=12, color='rgb(0,0,0)')

)

data = [trace]

fig = go.Figure(data = data, layout = layout)

iplot(fig)
# company wise status ratio



success_data = space_data[space_data['Status Mission']=='Success']

success_data.head()
# success rate for company



success_data_analysis_company = pd.DataFrame(success_data['Company Name'].value_counts().sort_values(ascending=False))

success_data_analysis_company = success_data_analysis_company.rename(columns={'Company Name':'Count'})

company_launch_analysis['Count2'] = success_data_analysis_company['Count']

company_launch_analysis['Success rate'] = (company_launch_analysis['Count2']/company_launch_analysis['Count'])*100

company_launch_analysis = company_launch_analysis.fillna(0)

success_rate_company = pd.DataFrame(company_launch_analysis['Success rate'].sort_values(ascending=False))
trace = go.Bar(x = success_rate_company.index,

              y = success_rate_company['Success rate'],

              marker = dict(color='rgba(120, 110, 180, 0.5)',

              line = dict(color='rgb(0,0,0)', width=1.5)))



layout = go.Layout(title="Success rate of various company",

                  xaxis=dict(title='Company Name',zeroline= False,

                         gridcolor='rgb(183,183,183)',showline=True),

                  yaxis=dict(title='Success rate(in percentage)',zeroline= False,

                            gridcolor='rgb(183,183,183)',showline=True),

                  font=dict(family='Courier New, monospace', size=12, color='rgb(0,0,0)')

)

data = [trace]

fig = go.Figure(data = data, layout = layout)

iplot(fig)
fig = px.pie(success_rate_company, values=success_rate_company['Success rate'], names=success_rate_company.index,

             title='Success Ratio of Companies in the world',

            )

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.update_layout(

    template='plotly_white'

)

fig.show()
#  success rate of country

success_data_analysis_country = pd.DataFrame(success_data['Country'].value_counts().sort_values(ascending=False))

success_data_analysis_country = success_data_analysis_country.rename(columns={'Country':'Count'})

country_launch_analysis['Count2'] = success_data_analysis_country['Count']

country_launch_analysis['Success rate'] = (country_launch_analysis['Count2']/country_launch_analysis['Count'])*100

country_launch_analysis = country_launch_analysis.fillna(0)

success_rate_country = pd.DataFrame(country_launch_analysis['Success rate'].sort_values(ascending=False))
trace = go.Bar(x = success_rate_country.index[:15],

              y = success_rate_country['Success rate'][:15],

              marker = dict(color='rgba(125, 215, 180, 0.5)',

              line = dict(color='rgb(0,0,0)', width=1.5)))



layout = go.Layout(title="Success rate of various countries",

                  xaxis=dict(title='Country Name',zeroline= False,

                         gridcolor='rgb(183,183,183)',showline=True),

                  yaxis=dict(title='Success rate(in percentage)',zeroline= False,

                            gridcolor='rgb(183,183,183)',showline=True),

                  font=dict(family='Courier New, monospace', size=12, color='rgb(0,0,0)')

)

data = [trace]

fig = go.Figure(data = data, layout = layout)

iplot(fig)
fig = px.pie(success_rate_country, values=success_rate_country['Success rate'], names=success_rate_country.index,

             title='Success Ratio of countries in The World',

            )

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.update_layout(

    template='plotly_white'

)

fig.show()
countries_dict = {

    'Russia' : 'Russian Federation',

    'New Mexico' : 'USA',

    "Yellow Sea": 'China',

    "Shahrud Missile Test Site": "Iran",

    "Pacific Missile Range Facility": 'USA',

    "Barents Sea": 'Russian Federation',

    "Gran Canaria": 'USA',

    "India": 'India',

}



space_data['Country'] = space_data['Country'].replace(countries_dict)
space_sunbrust = space_data.groupby(['Country', 'Company Name', 'Status Mission'])['Datum'].count().reset_index()

space_sunbrust.columns = ['country', 'company', 'status', 'count']

fig = px.sunburst(

    space_sunbrust, 

    path=[

        'country', 

        'company', 

        'status'

    ], 

    values='count', 

    title='Sunburst chart for all countries',

)

fig.show()
def wordcloud(string):

    wc = WordCloud(width=800,height=500,mask=None,random_state=21, max_font_size=110,stopwords=stop_words).generate(string)

    fig=plt.figure(figsize=(16,8))

    plt.axis('off')

    plt.imshow(wc)
stop_words=set(STOPWORDS)

country_string = " ".join(space_data['Country'])

company_string = " ".join(space_data['Company Name'])

detail_string = " ".join(space_data['Detail'])

location_string = " ".join(space_data['Location'])
# country wordcloud

wordcloud(country_string)
# cloud for company

wordcloud(company_string)
wordcloud(location_string)
wordcloud(detail_string)
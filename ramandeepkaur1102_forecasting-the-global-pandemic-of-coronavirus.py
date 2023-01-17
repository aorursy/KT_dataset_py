import numpy as np # linear algebra

import pandas as pd # data processing

from scipy.integrate import odeint

import seaborn as sns

import scipy.stats as stats

import plotly.express as px

import plotly.graph_objs as go

from plotly.offline import iplot, init_notebook_mode

import plotly.figure_factory as ff

from plotly import subplots

from plotly.subplots import make_subplots

import plotly.graph_objs as go

import matplotlib.pyplot as plt

%matplotlib inline

py.init_notebook_mode(connected=True)

import folium

import warnings

warnings.filterwarnings('ignore')



data = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', parse_dates=['Date'])

data = data.dropna(axis=0, how='all')

print(data.shape)

data.head(10)
for i in (data, data_copy):

    i['Year'] = i.Date.dt.year

    i['Month'] = i.Date.dt.month

    i['Day'] = i.Date.dt.day



data.head()
# Calculate ActiveCorona cases

data['Active'] = data['Confirmed'] - data['Deaths'] - data['Recovered']



data.head(10)



data_copy = data.copy()
# Bar plot for spread, death, recovered and active cases over the time around the world

covid_all = data.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

covid_all= covid_all[covid_all['Date'] > '2020-01-22']



# Plotting Values for Confirmed, deaths and reocvered cases

fig = make_subplots(rows=2, cols=2,specs=[[{}, {}],[{}, {}]],

                    subplot_titles=(f"{int(covid_all.Confirmed.max()):,d}" +' ' + "CONFIRMED", 

                    f"{int(covid_all.Recovered.max()):,d}" +' ' +"RECOVERED", 

                    f"{int(covid_all.Deaths.max()):,d}" +' ' +"DEATHS", 

                    f"{int(covid_all.Active.max()):,d}" +' ' +"ACTIVE"))



fig.add_trace(go.Bar(x=covid_all['Date'], y=covid_all['Confirmed'], text = covid_all['Confirmed'],

                     marker_color='Orange'), row=1, col=1)



fig.add_trace(go.Bar(x=covid_all['Date'], y=covid_all['Recovered'], marker_color='Green'), row=1, col=2)



fig.add_trace(go.Bar(x=covid_all['Date'], y=covid_all['Deaths'], marker_color='Red'), row=2, col=1)



fig.add_trace(go.Bar(x=covid_all['Date'], y=covid_all['Active'], marker_color='Blue'), row=2, col=2)



fig.update_traces(marker_line_color='rgb(8,48,107)',marker_line_width=1.5, opacity=0.8,

                  texttemplate='%{text:.2s}', textposition='inside')



fig['layout']['yaxis1'].update(title='Count', range=[0, covid_all['Confirmed'].max() + 15000])

fig['layout']['yaxis2'].update(title='Count', range=[0, covid_all['Recovered'].max() + 15000])

fig['layout']['yaxis3'].update(title='Count', range=[0, covid_all['Deaths'].max() + 15000])

fig['layout']['yaxis4'].update(title='Count', range=[0, covid_all['Active'].max() + 15000])

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')



fig.update_layout(template="ggplot2",title_text = '<b>Global COVID-19 Analysis</b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'), showlegend=False)

fig.show()
data.index = data['Date']

ts1 = data['Confirmed']

ts2 = data['Recovered']

ts3 = data['Active']

ts4 = data['Deaths']



plt.figure(figsize=(10, 6))

plt.plot(ts1, label='Confirmed', color='magenta', linewidth=2)

plt.plot(ts2, label='Recovered',color='cyan', linewidth=2)

plt.plot(ts3, label='Active', color='yellow', linewidth=2)

plt.plot(ts4, label='Deaths', color='red', linewidth=2)



plt.title('COVID-19 Patients Count Chart')



plt.xlabel("Time(Year-Month)")

plt.ylabel("Count")

plt.legend(loc='best')

plt.show()
data.groupby('Month')['Confirmed'].sum().plot.bar(color='magenta')

data.groupby('Month')['Active'].sum().plot.bar(color='cyan')

data.groupby('Month')['Recovered'].sum().plot.bar(color='yellow')

data.groupby('Month')['Deaths'].sum().plot.bar(color='red')
data1 = pd.read_csv('../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv')

data1.head()
data1 = data1[data1.columns[:-8]]

data1.drop('Unnamed: 3', axis=1, inplace=True)

data1['reporting date'] = pd.to_datetime(data1['reporting date'])

data1['exposure_start'] = pd.to_datetime(data1['exposure_start'])

data1['exposure_end'] = pd.to_datetime(data1['exposure_end'])

data1['hosp_visit_date'] = pd.to_datetime(data1['hosp_visit_date'])

data1['symptom_onset'] = pd.to_datetime(data1['symptom_onset'])



data1.head(2)
data2 = pd.read_csv('../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv')

data2 = data2[data2.columns[:-12]]

data2.loc[data2['sex']=='male', 'sex'] = 'Male'

data2.loc[data2['sex']=='female', 'sex'] = 'Female'

data2.head(2)
data_loc = pd.DataFrame(data1.groupby(['country'])['location'].nunique()).reset_index().sort_values(by='location', ascending=False).reset_index(drop=True)

data_loc.loc[data_loc.shape[0]]=['Total: '+str(data_loc['country'].nunique()), 'Total: '+str(data_loc['location'].sum())]

data_loc.head(10)
fig = px.pie(data1, values=[data1['gender'].value_counts()[0], data1['gender'].value_counts()[1]], names=['Male', 'Female'], title='Male v Female Affected Ratio')

fig.show()
fig = px.violin(data2[data2['sex']!='4000'].dropna(subset=['age', 'sex']), y="age", x='sex', color="sex",

                hover_data=data2.columns, title='Age Ratio of people affected b/w the two genders')

fig.show()
data1['sym_exp_diff'] = (data1['symptom_onset'] - data1['exposure_end']).dt.days

data1['hosp_sym_diff'] = (data1['hosp_visit_date'] - data1['symptom_onset']).dt.days
m = folium.Map(location=[0, 0], tiles='cartodbpositron',

               min_zoom=1, max_zoom=4, zoom_start=1)



data_mapping = data.dropna(subset=['Confirmed']).reset_index(drop=True)



for i in range(0, len(data_mapping)):

    folium.Circle(

        location=[data_mapping.iloc[i]['Lat'], data_mapping.iloc[i]['Long']],

        color='crimson', 

        tooltip =   '<li><bold>Country : '+str(data_mapping.iloc[i]['Country/Region'])+

                    '<li><bold>Province : '+str(data_mapping.iloc[i]['Province/State'])+

                    '<li><bold>Confirmed : '+str(data_mapping.iloc[i]['Confirmed'])+

                    '<li><bold>Deaths : '+str(data_mapping.iloc[i]['Deaths'])+

                    '<li><bold>Recovered : '+str(data_mapping.iloc[i]['Recovered']),

        radius=int(data_mapping.iloc[i]['Confirmed'])**1.1).add_to(m)

m
ncov_data = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')



ncov_data['ObservationDate'] = pd.to_datetime(ncov_data['ObservationDate']) 



ncov_data["Country"] = ncov_data["Country/Region"].replace(

    {

        "Mainland China": "China",

        "Hong Kong SAR": "Hong Kong",

        "Taipei and environs": "Taiwan",

        "Iran (Islamic Republic of)": "Iran",

        "Republic of Korea": "South Korea",

        "Republic of Ireland": "Ireland",

        "Macao SAR": "Macau",

        "Russian Federation": "Russia",

        "Republic of Moldova": "Moldova",

        "Taiwan*": "Taiwan",

        "Cruise Ship": "Others",

        "United Kingdom": "UK",

        "Viet Nam": "Vietnam",

        "Czechia": "Czech Republic",

        "St. Martin": "Saint Martin",

        "Cote d'Ivoire": "Ivory Coast",

        "('St. Martin',)": "Saint Martin",

        "Congo (Kinshasa)": "Congo",

    }

)

ncov_data["Province"] = ncov_data["Province/State"].fillna("-").replace(

    {

        "Cruise Ship": "Diamond Princess cruise ship",

        "Diamond Princess": "Diamond Princess cruise ship"

    }

)
ncov_data.head()
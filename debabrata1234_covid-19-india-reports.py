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
# import the necessary libraries

import numpy as np 
import pandas as pd
from datetime import date
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown
import plotly.graph_objs as go
import plotly.offline as py
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.offline import init_notebook_mode, plot, iplot, download_plotlyjs
import plotly as ply
import pycountry
import folium 
from folium import plugins
import json
from pandas.io.json import json_normalize


%config InlineBackend.figure_format = 'retina'
init_notebook_mode(connected=True)

# color pallette
cnf = '#393e46' # confirmed - grey
dth = '#ff2e63' # death - red
rec = '#21bf73' # recovered - cyan
act = '#fe9801' # active case - yellow

# India Latitude Longitude
India_Latitude = 21.7679
India_Longitude = 78.8718 
# Utility Functions

'''Display markdown formatted output like bold, italic bold etc.'''
def formatted_text(string):
    display(Markdown(string))


'''highlight the maximum in a Series or DataFrame'''  
def highlight_max(data, color='red'):
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''), index=data.index, columns=data.columns)
    


# Utility Plotting Functions

def plotDailyReportedCasesOverTime(df, country):
    # confirmed
    fig = px.bar(df, x="Date", y="Confirmed")
    layout = go.Layout(
        title=go.layout.Title(text="Daily count of confirmed cases in "+ country, x=0.5),
        font=dict(size=14),
        width=800,
        height=500,
        xaxis_title = "Date",
        yaxis_title = "Confirmed cases")

    fig.update_layout(layout)
    fig.show()

    # deaths
    fig = px.bar(df, x="Date", y="Deaths")
    layout = go.Layout(
        title=go.layout.Title(text="Daily count of reported deaths in "+ country, x=0.5),
        font=dict(size=14),
        width=800,
        height=500,
        xaxis_title = "Date",
        yaxis_title = "Deaths Reported")

    fig.update_layout(layout)
    fig.show()

    # recovered
    fig = px.bar(df, x="Date", y="Recovered")
    layout = go.Layout(
        title=go.layout.Title(text="Daily count of recovered cases in "+ country, x=0.5),
        font=dict(size=14),
        width=800,
        height=500,
        xaxis_title = "Date",
        yaxis_title = "Recovered Cases")

    fig.update_layout(layout)
    fig.show()
    
# Cases over time
def scatterPlotCasesOverTime(df, country):
    plot = make_subplots(rows=1, cols=3, subplot_titles=("Comfirmed", "Deaths", "Recovered"))

    subPlot1 = go.Scatter(
                    x=df['Date'],
                    y=df['Confirmed'],
                    name="Confirmed",
                    line_color='orange',
                    opacity=0.8)

    subPlot2 = go.Scatter(
                    x=df['Date'],
                    y=df['Deaths'],
                    name="Deaths",
                    line_color='red',
                    opacity=0.8)

    subPlot3 = go.Scatter(
                    x=df['Date'],
                    y=df['Recovered'],
                    name="Recovered",
                    line_color='green',
                    opacity=0.8)

    plot.append_trace(subPlot1, 1, 1)
    plot.append_trace(subPlot2, 1, 2)
    plot.append_trace(subPlot3, 1, 3)
    plot.update_layout(template="ggplot2", title_text = country + '<b> - Spread of the nCov Over Time</b>')

    plot.show()
covid_19_India = pd.read_csv("../input/covid19-in-india/covid_19_india.csv")
population_India_census2011 = pd.read_csv("../input/covid19-in-india/population_india_census2011.csv")

covid19_complete = pd.read_csv('../input/covid19-corona-virus-india-dataset/complete.csv', parse_dates=['Date'])

covid_19_India.head()
# covid_19_India['Confirmed'] = covid_19_India['ConfirmedIndianNational'] + covid_19_India['ConfirmedForeignNational']
covid_19_India.rename(columns={'State/UnionTerritory': 'State', 'Cured': 'Recovered'}, inplace=True)

covid_19_India.head()
covid_India_cases = pd.read_csv('../input/coronavirus-cases-in-india/Covid cases in India.csv')

# Coordinates of Indian States
India_Lat_Lon = pd.read_csv('../input/coronavirus-cases-in-india/Indian Coordinates.csv')

# Day by day data
dbd_India = pd.read_excel('../input/coronavirus-cases-in-india/per_day_cases.xlsx',sheet_name='India')
dbd_Italy = pd.read_excel('../input/coronavirus-cases-in-india/per_day_cases.xlsx',sheet_name="Italy")
dbd_Korea = pd.read_excel('../input/coronavirus-cases-in-india/per_day_cases.xlsx',sheet_name="Korea")

covid_India_cases.head()
covid_India_cases.rename(columns={'Name of State / UT': 'State', 'Cured/Discharged/Migrated': 'Recovered', 'Total Confirmed cases': 'Confirmed'}, inplace=True)

# covid_India_cases['Confirmed'] = covid_India_cases['Total Confirmed cases (Indian National)'] + covid_India_cases['Total Confirmed cases ( Foreign National )']

# Active Case = confirmed - deaths - recovered
covid_India_cases['Active'] = covid_India_cases['Confirmed'] - covid_India_cases['Deaths'] - covid_India_cases['Recovered']

covid_India_cases.style.background_gradient(cmap="Blues", subset=['Confirmed', 'Active'])\
            .background_gradient(cmap="Greens", subset=['Recovered'])\
            .background_gradient(cmap="Reds", subset=['Deaths'])
fig = px.bar(covid_India_cases.sort_values('Confirmed', ascending=False).sort_values('Confirmed', ascending=True), 
             x="Confirmed", y="State", title='Total Confirmed Cases', text='Confirmed', orientation='h', 
             width=16*(max(covid_India_cases['Confirmed']) + 2), height=700, range_x = [0, max(covid_India_cases['Confirmed']) + 2])
fig.update_traces(marker_color='#0726ed', opacity=0.8, textposition='outside')

fig.update_layout(plot_bgcolor='rgb(208, 236, 245)')
fig.show()
# fig = make_subplots(rows=1, cols=2, subplot_titles=("Indian Cases","Foreign Cases"))

# # Indian Nationals
# Indian = covid_India_cases.sort_values('Total Confirmed cases (Indian National)', ascending=False).sort_values('Total Confirmed cases (Indian National)', ascending=False)

# fig.add_trace(go.Bar( y=Indian['Total Confirmed cases (Indian National)'], x=Indian["State"],  
#                      marker=dict(color=Indian['Total Confirmed cases (Indian National)'], coloraxis="coloraxis")), 1, 1)

# # Foreign Nationals
# foreign = covid_India_cases.sort_values('Total Confirmed cases ( Foreign National )', ascending=False).sort_values('Total Confirmed cases ( Foreign National )', ascending=False)

# fig.add_trace(go.Bar( y=foreign['Total Confirmed cases ( Foreign National )'], x=foreign["State"], 
#                      marker=dict(color=foreign['Total Confirmed cases ( Foreign National )'], coloraxis="coloraxis")), 1, 2)                     
                     

# fig.update_layout(coloraxis=dict(colorscale='hsv'), showlegend=False,title_text="Indian vs Foreign Cases",plot_bgcolor='rgb(255, 255, 255)')
# fig.show()
covid_19_India.info()
# covid_19_India['Confirmed'] = covid_19_India['Confirmed'].replace('--', 0)
# covid_19_India['Confirmed'] = pd.to_numeric(covid_19_India['Confirmed'], errors='coerce', downcast='integer')

# covid_19_India.head()
temp = covid_19_India[["Date","Confirmed","Deaths","Recovered"]]
temp['Date'] = temp['Date'].apply(pd.to_datetime, dayfirst=True)

date_wise_data = temp.groupby(["Date"]).sum().reset_index()

formatted_text('***Day wise distribution for Confirmed, Deaths and Recovered Cases***')
# pd.set_option('display.max_rows', 100) 
date_wise_data
temp = date_wise_data.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Confirmed'],
                 var_name='Case', value_name='Count')
# temp.head()

fig = px.area(temp, x="Date", y="Count", color='Case',title='Cases over time', color_discrete_sequence = [rec, dth, act])
fig.show()
statewise_cases = pd.DataFrame(covid_India_cases.groupby(['State'])['Confirmed', 'Deaths', 'Recovered'].max().reset_index())
statewise_cases["Country"] = "India" # in order to have a single root node
fig = px.treemap(statewise_cases, path=['Country','State'], values='Confirmed',
                  color='Confirmed', hover_data=['State'],
                  color_continuous_scale='RdBu')
fig.show()
# Load the Indian state geo json file

# with open('../input/indian-state-geojson-data/india_state_geo.json') as file:
#     ind_geo = json.load(file)

# India_conf_choropleth = go.Figure(go.Choroplethmapbox(geojson=ind_geo, locations=Covid_India_With_Location['State'],
#                                                       z=Covid_India_With_Location['TotalConfirmed'], colorscale='Sunset',
#                                                       zmin=0, zmax=max(Covid_India_With_Location.TotalConfirmed), marker_opacity=0.5, marker_line_width=0))

# India_conf_choropleth.update_layout(mapbox_style="carto-positron", mapbox_zoom=3.5, 
#                                     mapbox_center = {"lat": India_Latitude, "lon": India_Longitude})

# India_conf_choropleth.update_layout(margin={"r":10,"t":0,"l":10,"b":0})

# iplot(India_conf_choropleth)


# Affected States

nCoV_States = covid_19_India['State'].unique().tolist()
print('\n')
print(nCoV_States)
print("\n------------------------------------------------------------------")
print("\nTotal States affected by nCoV: ",len(nCoV_States))
formatted_text('***Confirmed Cases vs Day***')
plt.figure(figsize=(16,6))
sns.barplot(x='Date',y='Confirmed',data=covid_19_India, order=covid_19_India.Date.unique().tolist())
plt.title('Distribution of total confirmed cases on every day basis')
plt.xticks(rotation=45)
# to plot the spread over time, we would need the data distribution spread over time starting from 22nd Jan 2020
# so we will extract the sub-set from the original data.

scatterPlotCasesOverTime(date_wise_data, "<b>India</b>")
# cases over time - confirmed vs deaths
plotDailyReportedCasesOverTime(date_wise_data, "INDIA")
covid19_complete.columns
covid19_complete.rename(columns={'Name of State / UT': 'State_UT', 'Cured/Discharged/Migrated': 'Recovered', 'Total Confirmed cases': 'Confirmed', 'Death': 'Deaths'}, inplace=True)

for i in ['Confirmed', 'Deaths', 'Recovered']:
    covid19_complete[i] = covid19_complete[i].astype('int')

# Derived Columns
covid19_complete['Active'] = covid19_complete['Confirmed'] - covid19_complete['Deaths'] - covid19_complete['Recovered']
covid19_complete['Mortality_rate(%)'] = covid19_complete['Deaths']/covid19_complete['Confirmed'] * 100
covid19_complete['Recovery_rate(%)'] = covid19_complete['Recovered']/covid19_complete['Confirmed'] * 100

covid19_complete = covid19_complete[['Date', 'State_UT', 'Latitude', 'Longitude', 'Confirmed', 'Active', 'Recovered', 'Deaths', 'Recovery_rate(%)', 'Mortality_rate(%)']]

covid19_complete.head()
# Get the latest data
latest_day = max(covid19_complete['Date'])
day_before = latest_day - timedelta(days = 1)

# state wise data and new cases reported in the last day
latest_day_data = covid19_complete[covid19_complete['Date']==latest_day].set_index('State_UT')
day_before_data = covid19_complete[covid19_complete['Date']==day_before].set_index('State_UT')

temp = pd.merge(left = latest_day_data, right = day_before_data, on='State_UT', suffixes=('_latest_day', '_previous_day'), how='outer')

# Get the number of new cases reported in the last day
latest_day_data['New_cases_reported'] = temp['Confirmed_latest_day'] - temp['Confirmed_previous_day']
latest = latest_day_data.reset_index()
latest.fillna(1, inplace=True)

temp = latest[['State_UT', 'Confirmed', 'Active', 'New_cases_reported','Recovered', 'Deaths', 'Recovery_rate(%)', 'Mortality_rate(%)']]
temp = temp.sort_values('Confirmed', ascending=False).reset_index(drop=True)

temp.style\
    .background_gradient(cmap="Blues", subset=['Confirmed', 'Active', 'New_cases_reported'])\
    .background_gradient(cmap="Greens", subset=['Recovered', 'Recovery_rate(%)'])\
    .background_gradient(cmap="Reds", subset=['Deaths', 'ortality_rate(%)'])
ordered_latest = latest.sort_values('Confirmed', ascending=False)
# state_order = temp['State/UT']

fig = px.bar(latest.sort_values('Confirmed', ascending=False), 
             x="Confirmed", y="State_UT", color='State_UT', title='Confirmed',
             orientation='h', text='Confirmed', height=700,
             color_discrete_sequence = px.colors.cyclical.IceFire)
fig.show()
date_wise_data = covid19_complete[["Date", 'State_UT', "Confirmed","Deaths","Recovered",'Active']]
date_wise_data['Date'] = date_wise_data['Date'].apply(pd.to_datetime, dayfirst=True)
date_wise_data = date_wise_data.groupby(["Date"]).sum().reset_index()
temp = date_wise_data.copy()
temp['Recovery_rate(%)'] = temp['Recovered']/temp['Confirmed']*100
temp['Mortality_rate(%)'] = temp['Deaths']/temp['Confirmed']*100
fig = go.Figure()
fig.add_trace(go.Scatter(x=temp['Date'], y=temp['Recovery_rate(%)'],
                    mode='lines+markers',marker_color='green'))
fig.update_layout(title_text = 'Cummulative Recovery Rate of India')
fig.update_layout(plot_bgcolor='rgb(275, 270, 273)',width=600, height=600)
fig.show()
fig = go.Figure()
fig.add_trace(go.Scatter(x=temp['Date'], y=temp['Mortality_rate(%)'],
                    mode='lines+markers',marker_color='red'))
fig.update_layout(title_text = 'Cummulative Mortality Rate of India')
fig.update_layout(plot_bgcolor='rgb(275, 270, 273)',width=600, height=600)
fig.show()
fig = go.Figure()
fig.add_trace(go.Scatter(x=date_wise_data['Date'], y=date_wise_data['Confirmed'],
                    mode='lines+markers',marker_color='blue',name='Total Cases'))

fig.add_trace(go.Scatter(x=date_wise_data['Date'],y=date_wise_data['Active'], 
                mode='lines+markers',marker_color='purple',name='Active'))

fig.add_trace(go.Scatter(x=date_wise_data['Date'], y=date_wise_data['Recovered'],
                mode='lines+markers',marker_color='green',name='Recovered'))

fig.add_trace(go.Scatter(x=date_wise_data['Date'], y=date_wise_data['Deaths'], 
                mode='lines+markers',marker_color='red',name='Deaths'))

fig.update_layout(title_text='Combined Weekly Trend - India',plot_bgcolor='rgb(275, 270, 273)',width=800, height=800)
fig.show()
agegroup_data = pd.read_csv('../input/covid19-in-india/AgeGroupDetails.csv')
agegroup_data.head()
fig = go.Figure()
fig.add_trace(go.Scatter(x=agegroup_data['AgeGroup'],y=agegroup_data['TotalCases'],line_shape='spline',fill='tonexty',fillcolor = 'orange')) 
fig.update_layout(title="Age wise Trend...",yaxis_title="Total Number of cases",xaxis_title="Age Group")
fig.update_layout(plot_bgcolor='rgb(275, 270, 273)',width=800,height=600)
fig.show()
temp = agegroup_data.copy()
temp['Percentage'] = temp['Percentage'].str.rstrip('%')
temp['Percentage'] = pd.to_numeric(temp['Percentage'])
temp.reset_index()
temp.set_index(["AgeGroup"], inplace = True, append = False, drop = True) 
temp

plt.figure(figsize=(12,12))

temp['Percentage'].plot( kind='pie'
           , autopct='%1.1f%%'
           , shadow=True
           , startangle=90)

plt.title('Agewise Distribution - INDIA',size=25)
plt.legend(loc = "best"
           , fontsize = 12
           , ncol = 1 
           , fancybox = True
           , framealpha = 0.80
           , shadow = True
           , borderpad = 1
           , bbox_to_anchor=(1,0.5));
url = 'https://api.covid19india.org/raw_data.json'

# Load the first sheet of the JSON file into a data frame
df = pd.read_json(url, orient='columns')
df1 = json_normalize(df['raw_data'])
df1.head()
covid19_patients = pd.read_csv('../input/covid19-corona-virus-india-dataset/patients_data.csv')
covid19_patients.head()
covid19_patients['date_announced'] = pd.to_datetime(covid19_patients['date_announced'], dayfirst=True)
covid19_patients['status_change_date'] = pd.to_datetime(covid19_patients['status_change_date'], dayfirst=True)
covid19_patients.info()
px.histogram(covid19_patients, x='age_bracket', color_discrete_sequence = ['#00ff95'], nbins=50, title='Age Distribution...')
covid19_patients.dropna(subset=['current_status', 'age_bracket'], inplace=True)
covid19_patients.reset_index(drop=True, inplace=True)

covid19_patients_deceased = covid19_patients[covid19_patients['current_status'] == 'Deceased']
covid19_patients_hospitalized = covid19_patients[covid19_patients['current_status'] == 'Hospitalized']
covid19_patients_recovered = covid19_patients[covid19_patients['current_status'] == 'Recovered']

fig = go.Figure()
fig.add_trace(go.Box(y=covid19_patients_deceased['age_bracket'], name="Deceased Patients"))
fig.add_trace(go.Box(y=covid19_patients_hospitalized['age_bracket'], name="Hospitalized Patients"))
fig.add_trace(go.Box(y=covid19_patients_recovered['age_bracket'], name="Recovered Patients"))
fig.update_layout(title_text='Indian COVID-19 Patients Outcome Age-Wise')
fig.show()
px.histogram(covid19_patients, x='gender', color_discrete_sequence = ['#00aeff'], title='Gender Distribution...')
temp = covid19_patients['gender'].value_counts().rename_axis('gender').reset_index(name='counts')

fig = make_subplots(
    rows=1, cols=1,
    subplot_titles = ['Gender Distribution...'],
    specs=[[{"type": "pie"}]]
)

fig.add_trace(go.Pie(values=temp.counts.tolist(), labels=['Male', 'Female'], marker_colors = ['#6a0572', '#39065a']),1,1)
fig.show()
px.histogram(covid19_patients, x='current_status', color_discrete_sequence = ['#00aeff'], title='Current Status...')
fig = make_subplots(
    rows=1, cols=2, column_widths=[0.7, 0.3],
    subplot_titles = ['Patients ratio', 'Current status'],
    specs=[[{"type": "histogram"}, {"type": "pie"}]]
)

temp = covid19_patients[['age_bracket', 'current_status']].dropna()

gen_grp = temp.groupby('current_status').count()

fig.add_trace(go.Pie(values=gen_grp.values.reshape(-1).tolist(), labels=['Deceased', 'Hospitalized', 'Recovered'], 
                     marker_colors = ['#fd0054', '#393e46', '#40a798'], hole=.3),1, 2)

fig.add_trace(go.Histogram(x=temp[temp['current_status']=='Deceased']['age_bracket'], nbinsx=50, name='Deceased', marker_color='#fd0054'), 1, 1)
fig.add_trace(go.Histogram(x=temp[temp['current_status']=='Recovered']['age_bracket'], nbinsx=50, name='Recovered', marker_color='#40a798'), 1, 1)
fig.add_trace(go.Histogram(x=temp[temp['current_status']=='Hospitalized']['age_bracket'], nbinsx=50, name='Hospitalized', marker_color='#393e46'), 1, 1)

fig.update_layout(showlegend=False)
fig.update_layout(barmode='stack')
fig.data[0].textinfo = 'label+text+value+percent'

fig.show()
temp = covid19_patients.groupby('nationality')['patient_number'].count().reset_index()
temp = temp.sort_values('patient_number')
# temp = temp[temp['nationality']!='India']
px.bar(temp, x='patient_number', y='nationality', orientation='h', text='patient_number', width=600,
       color_discrete_sequence = ['#eb4034'], title='Nationality - Indian Cases vs Foreign Cases...')
locations = covid19_patients.groupby(['detected_state', 'detected_district', 'detected_city'])['patient_number'].count().reset_index()
locations['country'] = 'India'
fig = px.treemap(locations, path=["country", "detected_state", "detected_district", "detected_city"], values="patient_number", height=700,
           title='State ---> District --> City', color_discrete_sequence = px.colors.qualitative.Prism)

fig.data[0].textinfo = 'label+text+value+percent entry+percent root'
fig.show()
days_to_status_change = covid19_patients[['date_announced', 'status_change_date', 'current_status']].dropna()

days_to_status_change = days_to_status_change[days_to_status_change['status_change_date'] != days_to_status_change['date_announced']]
days_to_status_change['days_to_status_change'] = days_to_status_change['status_change_date'] - days_to_status_change['date_announced']
days_to_status_change['days_to_status_change'] = days_to_status_change['days_to_status_change'].dt.days

days_to_recover = days_to_status_change[days_to_status_change['current_status']=='Recovered']
days_to_recover.head()
days_to_recover['days_to_status_change'].unique()
px.box(days_to_status_change, x="current_status", y="days_to_status_change", color='current_status')
# print(covid19_patients['notes'].value_counts())
print(covid19_patients['notes'].unique())
covid19_patients['notes'] = covid19_patients['notes'].replace('Details Awaited', 'Details awaited')
covid19_patients['notes'] = covid19_patients['notes'].replace('Travelled from Dubai, UAE', 'Travelled from Dubai')
covid19_patients['notes'] = covid19_patients['notes'].replace('attended religious event Tablighi Jamaat in delhi', 'Attended Delhi Religious Conference')
covid19_patients['notes'] = covid19_patients['notes'].replace('Travelled from London', 'Travelled from UK')
covid19_patients['notes'] = covid19_patients['notes'].replace('Travelled from Dubai.', 'Travelled from Dubai')
covid19_patients['notes'] = covid19_patients['notes'].replace('Visitd Delhi', 'Travelled from Delhi')


temp = pd.DataFrame(covid19_patients.groupby('notes')['notes'].count().sort_values(ascending=False))
temp.columns = ['count']
temp = temp.reset_index()
temp = temp[temp['notes']!='Details awaited']

temp
fig = px.bar(temp.head(15).sort_values('count', ascending=True), x='count', y='notes', orientation='h', text='count', width=1500,
       color_discrete_sequence = ['#3e78b5'], title='Patient History & Notes...')
fig.update_xaxes(title='')
fig.update_yaxes(title='')

fig.update_layout(
    margin=dict(l=20, r=50, t=40, b=20),
    paper_bgcolor="LightSteelBlue",)
India = folium.Map(location = [India_Latitude,India_Longitude], min_zoom=4, max_zoom=6, zoom_start=4.5, tiles = 'cartodbpositron')

for lat, long, confirmed, active, deaths, recovered, state in zip(latest['Latitude'],
                                                           latest['Longitude'],
                                                           latest['Confirmed'],
                                                           latest['Active'],
                                                           latest['Deaths'],
                                                           latest['Recovered'], 
                                                           latest['State_UT']):

    if (deaths == 0):
        folium.Marker(location=[lat, long]
                    , popup = ('<strong>nCov Numbers:</strong> ' + '<br>' + 
                               '<strong>State:</strong> ' + str(state).capitalize() + '<br>'
                               '<strong>Confirmed:</strong> ' + str(int(confirmed)) + '<br>'
                               '<strong>Active:</strong> ' + str(int(active)) + '<br>'
                               '<strong>Deaths:</strong> ' + str(int(deaths)) + '<br>'
                               '<strong>Recovered:</strong> ' + str(int(recovered)) + '<br>')
                    , icon=folium.Icon(color='darkblue',icon='info-sign'), color='rgb(55, 83, 109)'
                    , tooltip = str(state).capitalize(), fill_color='rgb(55, 83, 109)').add_to(India)

    else:
        folium.Marker(location=[lat, long]
                    , popup = ('<strong>nCov Numbers:</strong> ' + '<br>' + 
                               '<strong>State:</strong> ' + str(state).capitalize() + '<br>'
                               '<strong>Confirmed:</strong> ' + str(int(confirmed)) + '<br>'
                               '<strong>Active:</strong> ' + str(int(active)) + '<br>'
                               '<strong>Deaths:</strong> ' + str(int(deaths)) + '<br>'
                               '<strong>Recovered:</strong> ' + str(int(recovered)) + '<br>')
                    , icon=folium.Icon(color='red', icon='info-sign'), color='rgb(26, 118, 255)'
                    , tooltip = str(state).capitalize(), fill_color='rgb(26, 118, 255)').add_to(India)
        
India

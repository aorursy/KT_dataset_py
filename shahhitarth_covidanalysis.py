!pip install pyforest
from pyforest import *
import json
from datetime import timedelta
from urllib.request import urlopen
import warnings
warnings.filterwarnings('ignore')
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import plot, iplot, init_notebook_mode
init_notebook_mode(connected=True)
cnf, dth, rec, act = '#393e46', '#ff2e63', '#21bf73', '#fe9801'
clean = pd.read_csv('/kaggle/input/covid-data/covid_19_clean_complete.csv')
clean.head()
clean.shape
day_wise = pd.read_csv('/kaggle/input/covid-data/day_wise.csv')
day_wise['Date'] = pd.to_datetime(day_wise['Date'])
day_wise.head()
day_wise.shape
day_wise.loc[(day_wise.Date == '2020-04-22')]
grouped = pd.read_csv('/kaggle/input/covid-data/full_grouped.csv')
grouped['Date'] = pd.to_datetime(grouped['Date'])
grouped.head()
grouped.shape
country_wise = pd.read_csv('/kaggle/input/covid-data/country_wise_latest(1).csv')
country_wise.head()
country_wise.shape
world = pd.read_csv('/kaggle/input/covid-data/worldometer.csv')
world.head()
world.shape
temp = day_wise[['Date', 'Deaths', 'Recovered', 'Active']] .tail(1)
temp = temp.melt(id_vars = 'Date', value_vars=['Active','Deaths','Recovered'])
fig = px.treemap(temp, path=['variable'], values='value', height=225, color_discrete_sequence=[act, rec, dth])
fig.data[0].textinfo = 'label+text+value'
fig.show()
def plot_map(df, col, pal):
    df = df[df[col]>0]
    fig = px.choropleth(df, locations='Country/Region', locationmode='country names', color=col, hover_name='Country/Region', title=col, hover_data=[col], color_continuous_scale=pal)
    fig.show()
plot_map(country_wise, 'Confirmed', 'matter')
plot_map(country_wise, 'Deaths', 'matter')
plot_map(country_wise, 'Recovered', 'matter')
plot_map(country_wise, 'Deaths / 100 Cases', 'matter')
fig = px.choropleth(grouped, locations='Country/Region', color = np.log(grouped['Confirmed']), locationmode='country names', hover_name='Country/Region', animation_frame=grouped['Date'].dt.strftime('%Y-%m-%d'), title='Cases over time', color_continuous_scale=px.colors.sequential.matter)
fig.update(layout_coloraxis_showscale=False)
fig.show()
def plot_daywise(col, hue):
    fig = px.bar(day_wise, x='Date', y=col, width=700, color_discrete_sequence=[hue])
    fig.update_layout(title=col, xaxis_title='', yaxis_title='')
    fig.show()
def plot_daywise_line(col, hue):
    fig = px.line(day_wise, x='Date', y=col, width=700, color_discrete_sequence=[hue])
    fig.update_layout(title=col, xaxis_title='', yaxis_title='')
    fig.show()
temp = grouped.groupby('Date')['Recovered', 'Deaths', 'Active'].sum().reset_index()
temp = temp.melt(id_vars='Date', value_vars=['Recovered', 'Deaths', 'Active'], var_name='Case', value_name='Count')
temp.head()

fig = px.area(temp, x='Date', y='Count', color='Case', height=600, width=700, title='Cases over time', color_discrete_sequence=[rec, dth, act]) 
fig.update_layout(xaxis_rangeslider_visible=True)
fig.show()
plot_daywise('Confirmed', '#333333')
plot_daywise('Recovered', '#333333')
plot_daywise('New cases', '#333333')
temp = day_wise[['Date', 'Recovered', 'Active']]
temp = temp.melt(id_vars='Date', value_vars=['Recovered', 'Active'], var_name='Variable', value_name='Count')
px.line(temp, x='Date', y='Count', color='Variable')
def plot_hbar(df, col, n, hover_data=[]):
    fig = px.bar(df.sort_values(col).tail(n), x=col, y='Country/Region', color='WHO Region', text=col, orientation='h', width=700, hover_data=hover_data, color_discrete_sequence=px.colors.qualitative.Dark2)
    fig.update_layout(title=col, xaxis_title='', yaxis_title='', yaxis_categoryorder = 'total ascending', uniformtext_minsize=8, uniformtext_mode='hide')
    fig.show()
plot_hbar(country_wise, 'Confirmed', 15)
plot_hbar(country_wise, 'Active', 15)
plot_hbar(country_wise, 'Deaths', 15)
plot_hbar(country_wise, 'Deaths / 100 Cases', 15)
plot_hbar(country_wise, 'Recovered', 15)
plot_hbar(country_wise, '1 week change', 15)
fig = px.scatter(country_wise.sort_values('Deaths', ascending=False).iloc[:20, :], x='Confirmed', y='Deaths', color='Country/Region', size='Confirmed', height=700, text='Country/Region', log_x=True, log_y=True, title='Deaths vs Confirmed (Scale is in log10)')
fig.update_traces(textposition = 'top center')
fig.update_layout(showlegend = False)
fig.update_layout(xaxis_rangeslider_visible=True)
fig.show()
def plot_treemap(col):
    fig = px.treemap(country_wise, path=['Country/Region'], values=col, height=700, title=col, color_discrete_sequence=px.colors.qualitative.Dark2)
    fig.data[0].textinfo = 'label+text+value'
    fig.show()
plot_treemap('Confirmed')
plot_treemap('Recovered')
temp = grouped[['Date','Country/Region', 'New cases']]
temp['New cases reported ?'] = temp['New cases'] != 0
temp['New cases reported ?'] = temp['New cases reported ?'].astype(int)
fig = go.Figure(data=go.Heatmap(z=temp['New cases reported ?'], x=temp['Date'], y=temp['Country/Region'], colorscale='Emrld', showlegend=False, text=temp['New cases reported ?']))
fig.update_layout(yaxis = dict(dtick=1))
fig.update_layout(height=3000)
fig.show()
usa = pd.read_csv('/kaggle/input/covid-data/usa_county_wise.csv')
usa_latest = usa[usa['Date'] == max(usa['Date'])]
usa_grouped = usa_latest.groupby('Province_State')['Confirmed', 'Deaths'].sum().reset_index()
usa.head()
us_code = {'Alabama': 'AL', 'Alaska': 'AK', 'American Samoa': 'AS', 'Arizona': 'AZ', 'Arkansas': 'AR', 
    'California': 'CA','Colorado': 'CO','Connecticut': 'CT','Delaware': 'DE', 'District of Columbia': 'DC', 
    'Florida': 'FL', 'Georgia': 'GA', 'Guam': 'GU', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL',
    'Indiana': 'IN','Iowa': 'IA','Kansas': 'KS','Kentucky': 'KY','Louisiana': 'LA','Maine': 'ME',
    'Maryland': 'MD','Massachusetts': 'MA','Michigan': 'MI','Minnesota': 'MN','Mississippi': 'MS',
    'Missouri': 'MO','Montana': 'MT','Nebraska': 'NE','Nevada': 'NV','New Hampshire': 'NH', 'New Jersey': 'NJ',
    'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Northern Mariana Islands':'MP',
    'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Puerto Rico': 'PR',
    'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX',
    'Utah': 'UT', 'Vermont': 'VT', 'Virgin Islands': 'VI', 'Virginia': 'VA', 'Washington': 'WA',
    'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'}

usa_grouped['Code'] = usa_grouped['Province_State'].map(us_code)
fig = px.choropleth(usa_grouped, color='Confirmed', locations='Code', locationmode='USA-states', scope='usa', color_continuous_scale='RdGy', title='No. of cases in USA')
fig.show()
who = country_wise.groupby('WHO Region')['Confirmed', 'Deaths', 'Recovered', 'Active', 'New cases', 'Confirmed last week'].sum().reset_index()
who['Fatality Rate'] = round((who['Deaths']/who['Confirmed'])*100, 2)
who['Recovery Rate'] = (who['Recovered']/who['Confirmed'])*100
who_g = grouped.groupby(['WHO Region','Date'])['Confirmed', 'Deaths', 'Recovered', 'Active', 'New cases', 'New deaths'].sum().reset_index()
def plot_hbar(col, hover_data=[]):
    fig = px.bar(who.sort_values(col), x=col, y='WHO Region', text=col, orientation='h', width=700, hover_data=hover_data, color_discrete_sequence=px.colors.qualitative.Dark2)
    fig.update_layout(title=col, xaxis_title='', yaxis_title='', yaxis_categoryorder='total ascending', uniformtext_minsize=8, uniformtext_mode='hide')
    fig.show()
plot_hbar('Confirmed')
plot_hbar('Deaths')
plot_hbar('Recovered')
fig = px.scatter(country_wise, x='Confirmed', y='Deaths', color='WHO Region', height=700, hover_name='Country/Region', log_x=True, log_y=True, title='WHO Region wise', color_discrete_sequence=px.colors.qualitative.Vivid)
fig.update_traces(textposition='top center')
fig.show()
px.bar(who_g, x='Date', y='Confirmed', color='WHO Region', height=600, title='Confirmed', color_discrete_sequence=px.colors.qualitative.Vivid)
px.bar(who_g, x='Date', y='Confirmed', color='WHO Region', height=600, title='New cases', color_discrete_sequence=px.colors.qualitative.Vivid)
grouped['Week No.'] = grouped['Date'].dt.strftime('%U')
week_wise = grouped.groupby('Week No.')['Confirmed', 'Deaths', 'Recovered', 'Active', 'New cases', 'New deaths', 'New recovered'].sum().reset_index()
def plot_weekwise(col, hue):
    fig = px.bar(week_wise, x='Week No.', y=col, width=700, color_discrete_sequence=[hue])
    fig.update_layout(title=col, xaxis_title='', yaxis_title='')
    fig.show()
plot_weekwise('Confirmed', '#000000')
plot_weekwise('Deaths', dth)
plot_weekwise('New cases', '#cd6684')
grouped['Month'] = pd.DatetimeIndex(grouped['Date']).month
month_wise = grouped.groupby('Month')['Confirmed', 'Deaths', 'Recovered', 'Active', 'New cases', 'New deaths', 'New recovered'].sum().reset_index()
def plot_monthwise(col, hue):
    fig = px.bar(month_wise, x='Month', y=col, width=700, color_discrete_sequence=[hue])
    fig.update_layout(title=col, xaxis_title='', yaxis_title='')
    fig.show()
plot_monthwise('Confirmed', '#000000')
plot_monthwise('Recovered', '#cd6684')
temp = country_wise[country_wise['Active']==0]
temp = temp.sort_values('Confirmed', ascending=False)
temp.reset_index(drop=True)

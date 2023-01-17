import numpy as np

import pandas as pd



from datetime import datetime, timedelta, date, time



import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_dark"

from plotly.subplots import make_subplots



import matplotlib.pyplot as plt

import seaborn as sns



# hide warnings

import warnings

warnings.filterwarnings('ignore')

# color pallette

cnf = '#67000d' # confirmed - dark brown

dth = '#ff2e63' # death - red

rec = '#21bf73' # recovered - cyan

act = '#636efa' # active case - yellow
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as py

import plotly.graph_objs as go



init_notebook_mode(connected=True) #do not miss this line



data = [go.Bar(

        x=["Monday", "Tuesday"],

        y=[55,100]  )]

fig = go.Figure(data=data)



py.offline.iplot(fig)
ts19confirmed = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

ts19recover = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

ts19deaths = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
# confirmed = pd.melt(ts19confirmed, id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name='Date', value_name='Confirmed')

# recovered = pd.melt(ts19recover, id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name='Date', value_name='Recovered')

# deaths = pd.melt(ts19deaths, id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name='Date', value_name='Deaths')



# result = confirmed

# result['Deaths'] = deaths['Deaths'].values

# result['Recovered'] = recovered['Recovered'].values



# new_data = result

# new_data['Date'] = pd.to_datetime(new_data['Date'])

# new_data = new_data.reset_index(drop=True)

# new_data['Active'] = new_data['Confirmed'] - (new_data['Deaths'] + new_data['Recovered'])

# data = new_data

# without_china = data[data['Country/Region'] != 'China']
data = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv', parse_dates=['Date'])

data['Active'] = data['Confirmed'] - (data['Deaths'] + data['Recovered'])

# data = data[data['Date'] < '2020-03-23']

data = data

without_china = data[data['Country/Region'] != 'China']
# data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv', parse_dates=['ObservationDate', 'Last Update'])

# data = data.copy()

# data = data.drop(['SNo', 'Last Update'], axis=1)

# data['Country/Region'] = data['Country/Region'].replace('Mainland China', 'China')

# data = data.rename(columns={"ObservationDate": "Date"})

# data['Active'] = data['Confirmed'] - (data['Deaths'] + data['Recovered'])

# data.sort_values(by='Confirmed', ascending=False

# without_china = data[data['Country/Region'] != 'China']
last_max_data = data.groupby(['Date'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum()

last_max_data = last_max_data.reset_index()

last_max_data = last_max_data[last_max_data['Date'] == max(last_max_data['Date'])]

last_max_data = last_max_data.reset_index(drop=True)

last_max_data['Deaths %'] = round(100 * last_max_data['Deaths'] / last_max_data['Confirmed'], 2)

last_max_data['Recovered %'] = round(100 * last_max_data['Recovered'] / last_max_data['Confirmed'], 2)

last_max_data['Active %'] = round(100 * last_max_data['Active'] / last_max_data['Confirmed'], 2)
lencov19 = len(data['Country/Region'].unique())

epidemics = pd.DataFrame({

    'epidemic' : ['COVID-19', 'SARS', 'EBOLA', 'MERS', 'Swine flu'],

    'start_year' : [2019, 2002, 2014, 2012, 2009],

    'end_year' : [2020, 2004, 2016, 2020, 2010],

    'confirmed' : [last_max_data['Confirmed'].sum(), 8096, 28646, 2519, 6724149],

    'deaths' : [last_max_data['Deaths'].sum(), 774, 11323, 866, 19654],

    'total countries' : [lencov19, 29, 10, 27, 60],

    'began from': ['China', 'China', 'Guinea', 'Saudi Arabia', 'US'],

})



epidemics['mortality'] = round((epidemics['deaths']/epidemics['confirmed'])*100, 2)



epidemics.style.background_gradient(cmap='Pastel1')
temp = epidemics.melt(id_vars='epidemic', value_vars=['confirmed', 'deaths', 'mortality'],

                      var_name='Case', value_name='Value')



fig = px.bar(temp, x="epidemic", y="Value", color='epidemic', text='Value', facet_col="Case",

             color_discrete_sequence = px.colors.qualitative.Bold)

fig.update_traces(textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.update_yaxes(showticklabels=False)

fig.layout.yaxis2.update(matches=None)

fig.layout.yaxis3.update(matches=None)

fig.show()
last_max_data.style.background_gradient(cmap='Pastel1')
wc_last_max_data = without_china.groupby(['Date'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum()

wc_last_max_data = wc_last_max_data.reset_index()

wc_last_max_data = wc_last_max_data[wc_last_max_data['Date'] == max(wc_last_max_data['Date'])]

wc_last_max_data = wc_last_max_data.reset_index(drop=True)

wc_last_max_data['Deaths Rate'] = round(100 * wc_last_max_data['Deaths'] / wc_last_max_data['Confirmed'], 2)

wc_last_max_data['Recovered Rate'] = round(100 * wc_last_max_data['Recovered'] / wc_last_max_data['Confirmed'], 2)

wc_last_max_data['Active Rate'] = round(100 * wc_last_max_data['Active'] / wc_last_max_data['Confirmed'], 2)

wc_last_max_data.style.background_gradient(cmap='Pastel1')
def per_day_data_mrg(country_name):

    cr_per_day_data = data.groupby(['Date', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered'].sum()

    cr_per_day_data = cr_per_day_data.reset_index()

    cr_per_day_data = cr_per_day_data[cr_per_day_data['Country/Region'] == country_name]

    cr_per_day_data = cr_per_day_data.drop(['Country/Region'], axis=True).reset_index(drop=True)

    cr_per_day_data = cr_per_day_data.set_index(['Date']).diff()

    cr_per_day_data = cr_per_day_data.reset_index()

    cr_per_day_data.insert(1, 'Country/Region', country_name)

    cr_per_day_data[['Confirmed', 'Deaths', 'Recovered']] = cr_per_day_data[['Confirmed', 'Deaths', 'Recovered']].fillna(0)

    return cr_per_day_data



unq_country_name = data['Country/Region'].unique()

data_frames = []

for name in unq_country_name:

    get_result = per_day_data_mrg(name)

    data_frames.append(get_result)
cr_per_day_data_result = pd.concat(data_frames)

cr_per_day_data_result.reset_index()

num = cr_per_day_data_result._get_numeric_data()

num[num < 0] = 0.0

cr_per_day_data_result = cr_per_day_data_result.sort_values(by=['Date', 'Confirmed'], ascending=False).reset_index(drop=True)

cr_per_day_data_result.head().style.background_gradient(cmap='Reds')
cr_per_day_data_result = pd.concat(data_frames)

cr_per_day_data_result.reset_index()

num = cr_per_day_data_result._get_numeric_data()

num[num < 0] = 0.0

cr_per_day_data_result = cr_per_day_data_result.sort_values(by=['Date', 'Deaths'], ascending=False).reset_index(drop=True)

cr_per_day_data_result.head().style.background_gradient(cmap='Reds')
cr_per_day_data_result = pd.concat(data_frames)

cr_per_day_data_result.reset_index()

num = cr_per_day_data_result._get_numeric_data()

num[num < 0] = 0.0

cr_per_day_data_result = cr_per_day_data_result.sort_values(by=['Date', 'Recovered'], ascending=False).reset_index(drop=True)

cr_per_day_data_result.head().style.background_gradient(cmap='Greens')
ds_data = data.groupby('Date')['Confirmed', 'Deaths', 'Recovered'].sum().diff()

ds_data = ds_data.reset_index()

ds_data = ds_data.melt(id_vars="Date",value_vars=['Confirmed', 'Deaths', 'Recovered'])



fig = px.bar(ds_data, x="Date", y="value", color='variable', title='Daily Cases Whole world', color_discrete_sequence=['#536DFE', dth, rec])

fig.update_layout(barmode='group')

fig.show()



# -----------------outside china--------------------------



wc_ds_data = without_china.groupby('Date')['Confirmed', 'Deaths', 'Recovered'].sum().diff()

wc_ds_data = wc_ds_data.reset_index()

wc_ds_data = wc_ds_data.melt(id_vars="Date",value_vars=['Confirmed', 'Deaths', 'Recovered'])



fig = px.bar(wc_ds_data, x="Date", y="value", color='variable', title='Daily Cases Outside China', color_discrete_sequence=['#536DFE', dth, rec])

fig.update_layout(barmode='group')

fig.show()



# -----------------in us--------------------------

us_ds_data = data.copy()

us_ds_data = us_ds_data[us_ds_data['Country/Region'] == 'US']



confirmed_ds_data = us_ds_data[us_ds_data['Confirmed'] > 0]

date_time = confirmed_ds_data.values

date_time = date_time[0, 4] + pd.DateOffset(-2)

us_ds_data = us_ds_data[us_ds_data['Date'] > date_time]



us_ds_data = us_ds_data.groupby('Date')['Confirmed', 'Deaths', 'Recovered'].sum().diff()

us_ds_data = us_ds_data.reset_index()

us_ds_data = us_ds_data.melt(id_vars="Date",value_vars=['Confirmed', 'Deaths', 'Recovered'])



fig = px.bar(us_ds_data, x="Date", y="value", color='variable', title='Daily Cases In US', color_discrete_sequence=['#536DFE', dth, rec])

fig.update_layout(barmode='group')

fig.show()





# -----------------in italy--------------------------

it_ds_data = data.copy()

it_ds_data = it_ds_data[it_ds_data['Country/Region'] == 'Italy']



confirmed_ds_data = it_ds_data[it_ds_data['Confirmed'] > 0]

date_time = confirmed_ds_data.values

date_time = date_time[0, 4] + pd.DateOffset(-2)

it_ds_data = it_ds_data[it_ds_data['Date'] > date_time]



it_ds_data = it_ds_data.groupby('Date')['Confirmed', 'Deaths', 'Recovered'].sum().diff()

it_ds_data = it_ds_data.reset_index()

it_ds_data = it_ds_data.melt(id_vars="Date",value_vars=['Confirmed', 'Deaths', 'Recovered'])



fig = px.bar(it_ds_data, x="Date", y="value", color='variable', title='Daily Cases In Italy', color_discrete_sequence=['#536DFE', dth, rec])

fig.update_layout(barmode='group')

fig.show()



# -----------------in spain--------------------------

sp_ds_data = data.copy()

sp_ds_data = sp_ds_data[sp_ds_data['Country/Region'] == 'Spain']



confirmed_ds_data = sp_ds_data[sp_ds_data['Confirmed'] > 0]

date_time = confirmed_ds_data.values

date_time = date_time[0, 4] + pd.DateOffset(-2)

sp_ds_data = sp_ds_data[sp_ds_data['Date'] > date_time]



sp_ds_data = sp_ds_data.groupby('Date')['Confirmed', 'Deaths', 'Recovered'].sum().diff()

sp_ds_data = sp_ds_data.reset_index()

sp_ds_data = sp_ds_data.melt(id_vars="Date",value_vars=['Confirmed', 'Deaths', 'Recovered'])



fig = px.bar(sp_ds_data, x="Date", y="value", color='variable', title='Daily Cases In Spain', color_discrete_sequence=['#536DFE', dth, rec])

fig.update_layout(barmode='group')

fig.show()



# -----------------in china--------------------------

ic_ds_data = data.copy()

ic_ds_data = ic_ds_data[ic_ds_data['Country/Region'] == 'China']

ic_ds_data = ic_ds_data.groupby('Date')['Confirmed', 'Deaths', 'Recovered'].sum().diff()

ic_ds_data = ic_ds_data.reset_index()

ic_ds_data = ic_ds_data.melt(id_vars="Date",value_vars=['Confirmed', 'Deaths', 'Recovered'])



fig = px.bar(ic_ds_data, x="Date", y="value", color='variable', title='Daily Cases In China', color_discrete_sequence=['#536DFE', dth, rec])

fig.update_layout(barmode='group')

fig.show()



# -----------------in iran--------------------------

ir_ds_data = data.copy()

ir_ds_data = ir_ds_data[ir_ds_data['Country/Region'] == 'Iran']



confirmed_ds_data = ir_ds_data[ir_ds_data['Confirmed'] > 0]

date_time = confirmed_ds_data.values

date_time = date_time[0, 4] + pd.DateOffset(-2)

ir_ds_data = ir_ds_data[ir_ds_data['Date'] > date_time]



ir_ds_data = ir_ds_data.groupby('Date')['Confirmed', 'Deaths', 'Recovered'].sum().diff()

ir_ds_data = ir_ds_data.reset_index()

ir_ds_data = ir_ds_data.melt(id_vars="Date",value_vars=['Confirmed', 'Deaths', 'Recovered'])



fig = px.bar(ir_ds_data, x="Date", y="value", color='variable', title='Daily Cases In Iran', color_discrete_sequence=['#536DFE', dth, rec])

fig.update_layout(barmode='group')

fig.show()



# -----------------in india--------------------------

id_ds_data = data.copy()

id_ds_data = id_ds_data[id_ds_data['Country/Region'] == 'India']



confirmed_ds_data = id_ds_data[id_ds_data['Confirmed'] > 0]

date_time = confirmed_ds_data.values

date_time = date_time[0, 4] + pd.DateOffset(-2)

id_ds_data = id_ds_data[id_ds_data['Date'] > date_time]



id_ds_data = id_ds_data.groupby('Date')['Confirmed', 'Deaths', 'Recovered'].sum().diff()

id_ds_data = id_ds_data.reset_index()

id_ds_data = id_ds_data.melt(id_vars="Date",value_vars=['Confirmed', 'Deaths', 'Recovered'])



fig = px.bar(id_ds_data, x="Date", y="value", color='variable', title='Daily Cases In India', color_discrete_sequence=['#536DFE', dth, rec])

fig.update_layout(barmode='group')

fig.show()



# -----------------in bangladesh--------------------------

bd_ds_data = data.copy()

bd_ds_data = bd_ds_data[bd_ds_data['Country/Region'] == 'Bangladesh']



confirmed_ds_data = bd_ds_data[bd_ds_data['Confirmed'] > 0]

date_time = confirmed_ds_data.values

# date_time[0, 4]

date_time = date_time[0, 4] + pd.DateOffset(-2)

bd_ds_data = bd_ds_data[bd_ds_data['Date'] > date_time]



bd_ds_data = bd_ds_data.groupby('Date')['Confirmed', 'Deaths', 'Recovered'].sum().diff()

bd_ds_data = bd_ds_data.reset_index()

bd_ds_data = bd_ds_data.melt(id_vars="Date",value_vars=['Confirmed', 'Deaths', 'Recovered'])



fig = px.bar(bd_ds_data, x="Date", y="value", color='variable', title='Daily Cases In Bangladesh', color_discrete_sequence=['#536DFE', dth, rec])

fig.update_layout(barmode='group')

fig.show()
pi_data = last_max_data.melt(id_vars="Date", value_vars=['Active', 'Deaths', 'Recovered'], var_name='Case', value_name='Count')

fig = px.pie(pi_data, values='Count', names='Case', color_discrete_sequence=[act, rec, dth])

fig.show()
pi_data = wc_last_max_data.melt(id_vars="Date", value_vars=['Active', 'Deaths', 'Recovered'], var_name='Case', value_name='Count')

# df = px.pi_data.tips()

fig = px.pie(pi_data, values='Count', names='Case', color_discrete_sequence=[act, rec, dth])

fig.show()
country_data = data[data['Confirmed'] > 0]

country_data = country_data.groupby(['Date','Country/Region'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum()

country_data = country_data.reset_index()

country_data = country_data.drop_duplicates(subset=["Country/Region"], keep='last')

country_data = country_data.sort_values(by='Deaths', ascending=False)

country_data = country_data.reset_index(drop=True)

country_data['Deaths %'] = round(100 * country_data['Deaths'] / country_data['Confirmed'], 2)

country_data['Recovered %'] = round(100 * country_data['Recovered'] / country_data['Confirmed'], 2)

country_data['Active %'] = round(100 * country_data['Active'] / country_data['Confirmed'], 2)

country_data.style.background_gradient(cmap='Reds')
wc_country_data = without_china.groupby(['Date','Country/Region'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum()

wc_country_data = wc_country_data.reset_index()

wc_country_data = wc_country_data.drop_duplicates(subset=["Country/Region"], keep='last')

wc_country_data = wc_country_data.sort_values(by='Deaths', ascending=False)

wc_country_data = wc_country_data.reset_index(drop=True)

wc_country_data['Deaths Rate'] = round(100 * wc_country_data['Deaths'] / wc_country_data['Confirmed'], 2)

wc_country_data['Recovered Rate'] = round(100 * wc_country_data['Recovered'] / wc_country_data['Confirmed'], 2)

wc_country_data['Active Rate'] = round(100 * wc_country_data['Active'] / wc_country_data['Confirmed'], 2)

wc_country_data.style.background_gradient(cmap='Reds')
line_data = data.groupby(['Date'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

line_data = line_data.melt(id_vars="Date", value_vars=['Confirmed', 'Active', 'Recovered', 'Deaths'], var_name='Case', value_name='Count')

fig = px.line(line_data, x='Date', y='Count', color='Case', title='Whole World Cases over time', color_discrete_sequence = [cnf, act, rec, dth])

fig.show()
wc_line_data = without_china.groupby(['Date'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

wc_line_data = wc_line_data.melt(id_vars="Date", value_vars=['Confirmed', 'Active', 'Recovered', 'Deaths'], var_name='Case', value_name='Count')

fig = px.line(wc_line_data, x='Date', y='Count', color='Case', title='Outside China Cases over time', color_discrete_sequence = [cnf, act, rec, dth])

fig.show()
ch_data = data[data['Country/Region'] == 'China'].reset_index(drop=True)

ch_line_data = ch_data.groupby(['Date'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

ch_line_data = ch_line_data.melt(id_vars="Date", value_vars=['Confirmed', 'Active', 'Recovered', 'Deaths'], var_name='Case', value_name='Count')

fig = px.line(ch_line_data, x='Date', y='Count', color='Case', title='China Cases over time', color_discrete_sequence = [cnf, act, rec, dth])

fig.show()
it_data = data[data['Country/Region'] == 'Italy'].reset_index(drop=True)

it_data = it_data[it_data['Confirmed'] > 0]

it_line_data = it_data.groupby(['Date'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

it_line_data = it_line_data.melt(id_vars="Date", value_vars=['Confirmed', 'Active', 'Recovered', 'Deaths'], var_name='Case', value_name='Count')

fig = px.line(it_line_data, x='Date', y='Count', color='Case', title='Italy Cases over time', color_discrete_sequence = [cnf, act, rec, dth])

fig.show()
ir_data = data[data['Country/Region'] == 'Iran'].reset_index(drop=True)

ir_data = ir_data[ir_data['Confirmed'] > 0]

ir_line_data = ir_data.groupby(['Date'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

ir_line_data = ir_line_data.melt(id_vars="Date", value_vars=['Confirmed', 'Active', 'Recovered', 'Deaths'], var_name='Case', value_name='Count')

fig = px.line(ir_line_data, x='Date', y='Count', color='Case', title='Iran Cases over time', color_discrete_sequence = [cnf, act, rec, dth])

fig.show()
us_data = data[data['Country/Region'] == 'US'].reset_index(drop=True)

us_data = us_data[us_data['Confirmed'] > 0]

us_line_data = us_data.groupby(['Date'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

us_line_data = us_line_data.melt(id_vars="Date", value_vars=['Confirmed', 'Active', 'Recovered', 'Deaths'], var_name='Case', value_name='Count')

fig = px.line(us_line_data, x='Date', y='Count', color='Case', title='US Cases over time', color_discrete_sequence = [cnf, act, rec, dth])

fig.show()
in_data = data[data['Country/Region'] == 'India'].reset_index(drop=True)

in_data = in_data[in_data['Confirmed'] > 0]

in_line_data = in_data.groupby(['Date'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

in_line_data = in_line_data.melt(id_vars="Date", value_vars=['Confirmed', 'Active', 'Recovered', 'Deaths'], var_name='Case', value_name='Count')

fig = px.line(in_line_data, x='Date', y='Count', color='Case', title='India Cases over time', color_discrete_sequence = [cnf, act, rec, dth])

fig.show()
bd_data = data[data['Country/Region'] == 'Bangladesh'].reset_index(drop=True)

bd_data = bd_data[bd_data['Confirmed'] > 0]

bd_line_data = bd_data.groupby(['Date'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

bd_line_data = bd_line_data.melt(id_vars="Date", value_vars=['Confirmed', 'Active', 'Recovered', 'Deaths'], var_name='Case', value_name='Count')

fig = px.line(bd_line_data, x='Date', y='Count', color='Case', title='Bangladesh Cases over time', color_discrete_sequence = [cnf, act, rec, dth])

fig.show()
area_data = data.groupby(['Date'])['Deaths', 'Recovered', 'Active'].sum().reset_index()

area_data = area_data.melt(id_vars="Date", value_vars=['Deaths', 'Recovered', 'Active'], var_name='Case', value_name='Count')

fig = px.area(area_data, x="Date", y="Count", color='Case',

             title='Whole world Cases over time', color_discrete_sequence = [dth, rec, act])

fig.show()
wc_area_data = without_china.groupby(['Date'])['Deaths', 'Recovered', 'Active'].sum().reset_index()

wc_area_data = wc_area_data.melt(id_vars="Date", value_vars=['Deaths', 'Recovered', 'Active'], var_name='Case', value_name='Count')

fig = px.area(wc_area_data, x="Date", y="Count", color='Case',

             title='Outside China Cases over time', color_discrete_sequence = [dth, rec, act])

fig.show()
bar_data = data.groupby(['Date', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

bar_data = bar_data.drop_duplicates(subset=["Country/Region"], keep='last')
conf_bar_data = bar_data.sort_values(by='Confirmed', ascending=False).head(10)

conf_bar_data = conf_bar_data.sort_values(by='Confirmed', ascending=True)

fig = px.bar(conf_bar_data, x="Confirmed", y="Country/Region", title='Confirmed Cases for 10 Countries', text='Confirmed', orientation='h', 

             width=700, height=700, range_x = [0, max(bar_data['Confirmed'])+10000])

fig.update_traces(marker_color='#46cdcf', opacity=0.8, textposition='outside')

fig.show()











dth_bar_data = bar_data.sort_values(by='Deaths', ascending=False).head(10)

dth_bar_data = dth_bar_data.sort_values(by='Deaths', ascending=True)

fig = px.bar(dth_bar_data, x="Deaths", y="Country/Region", title='Deaths Cases for 10 Countries', text='Deaths', orientation='h', 

             width=700, height=700, range_x = [0, max(bar_data['Deaths'])+500])

fig.update_traces(marker_color='#ff2e63', opacity=0.8, textposition='outside')

fig.show()









rec_bar_data = bar_data.sort_values(by='Recovered', ascending=False).head(10)

rec_bar_data = rec_bar_data.sort_values(by='Recovered', ascending=True)

fig = px.bar(rec_bar_data, x="Recovered", y="Country/Region", title='Recovered Cases for 10 Countries', text='Recovered', orientation='h', 

             width=700, height=700, range_x = [0, max(bar_data['Recovered'])+10000])

fig.update_traces(marker_color='#21bf73', opacity=0.8, textposition='outside')

fig.show()







act_bar_data = bar_data.sort_values(by='Active', ascending=False).head(10)

act_bar_data = act_bar_data.sort_values(by='Active', ascending=True)

fig = px.bar(act_bar_data, x="Active", y="Country/Region", title='Active Cases for 10 Countries', text='Active', orientation='h', 

             width=700, height=700, range_x = [0, max(bar_data['Active'])+10000])

fig.update_traces(marker_color='#fe9801', opacity=0.8, textposition='outside')

fig.show()
wc_bar_data = without_china.groupby(['Date', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

wc_bar_data = wc_bar_data.drop_duplicates(subset=["Country/Region"], keep='last')



wc_bar_data['Deaths Rate'] = round(100 * wc_bar_data['Deaths'] / wc_bar_data['Confirmed'], 2)

wc_bar_data['Recovered Rate'] = round(100 * wc_bar_data['Recovered'] / wc_bar_data['Confirmed'], 2)

wc_bar_data['Active Rate'] = round(100 * wc_bar_data['Active'] / wc_bar_data['Confirmed'], 2)
wc_bar_data = wc_bar_data[wc_bar_data['Confirmed'] > 1000]

wc_daths_rate = wc_bar_data

wc_daths_rate = wc_daths_rate.sort_values(by=['Deaths Rate'], ascending=False).reset_index(drop=True)

wc_daths_rate.style.background_gradient(cmap='Reds')
wc_bar_data = wc_bar_data[wc_bar_data['Confirmed'] > 1000]

wc_dth_bar_data = wc_bar_data.sort_values(by='Deaths Rate', ascending=False)

wc_dth_bar_data = wc_dth_bar_data.sort_values(by='Deaths Rate', ascending=True)

fig = px.bar(wc_dth_bar_data, x="Deaths Rate", y="Country/Region", title='Outside China Minimum 1000 confirmed case Deaths Rate', text='Deaths Rate', orientation='h', 

             width=700, height=700, range_x = [0, max(wc_bar_data['Deaths Rate'])+5])

fig.update_traces(marker_color='#ff2e63', opacity=0.8, textposition='outside')

fig.show()







wc_rec_bar_data = wc_bar_data[wc_bar_data['Confirmed'] > 1000]

wc_rec_bar_data = wc_rec_bar_data.sort_values(by='Recovered Rate', ascending=False)

wc_rec_bar_data = wc_rec_bar_data.sort_values(by='Recovered Rate', ascending=True)

fig = px.bar(wc_rec_bar_data, x="Recovered Rate", y="Country/Region", title='Outside China Minimum 1000 confirmed case Recovered Rate', text='Recovered Rate', orientation='h', 

             width=700, height=700, range_x = [0, max(wc_bar_data['Recovered Rate'])+5])

fig.update_traces(marker_color='#21bf73', opacity=0.8, textposition='outside')

fig.show()









wc_act_bar_data = wc_bar_data[wc_bar_data['Confirmed'] > 1000]

wc_act_bar_data = wc_act_bar_data.sort_values(by='Active Rate', ascending=False)

wc_act_bar_data = wc_act_bar_data.sort_values(by='Active Rate', ascending=True)

fig = px.bar(wc_act_bar_data, x="Active Rate", y="Country/Region", title='Outside China Minimum 1000 confirmed case Active Rate', text='Active Rate', orientation='h', 

             width=700, height=700, range_x = [0, max(wc_bar_data['Active Rate'])+50])

fig.update_traces(marker_color='#fe9801', opacity=0.8, textposition='outside')

fig.show()
vertical_data = data.copy()

bar_ver_data = vertical_data.groupby(['Date', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum()

bar_ver_data = bar_ver_data.reset_index()

bar_ver_data = bar_ver_data.drop_duplicates(subset=["Country/Region"], keep='last')

bar_ver_data = bar_ver_data.sort_values(by='Confirmed', ascending=False).reset_index(drop=True)

bar_ver_data = bar_ver_data[bar_ver_data['Confirmed'] > 500]



bar_ver_data = bar_ver_data.melt(id_vars="Country/Region", value_vars=['Active', 'Recovered', 'Deaths'])



fig = px.bar(bar_ver_data.sort_values(['variable', 'value']), 

             x="Country/Region", y="value", color='variable', orientation='v', height=800,

             title='Minimum 500 confirmed cases, Whole world', color_discrete_sequence=[act, dth, rec])

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
wc_vertical = without_china.copy()

wc_bar_ver_data = wc_vertical.groupby(['Date', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum()

wc_bar_ver_data = wc_bar_ver_data.reset_index()

wc_bar_ver_data = wc_bar_ver_data.drop_duplicates(subset=["Country/Region"], keep='last')

wc_bar_ver_data = wc_bar_ver_data.sort_values(by='Confirmed', ascending=False).reset_index(drop=True)

wc_bar_ver_data = wc_bar_ver_data[wc_bar_ver_data['Confirmed'] > 500]



wc_bar_ver_data = wc_bar_ver_data.melt(id_vars="Country/Region", value_vars=['Active', 'Recovered', 'Deaths'])



fig = px.bar(wc_bar_ver_data.sort_values(['variable', 'value']), 

             x="Country/Region", y="value", color='variable', orientation='v', height=800,

             title='Minimum 500 confirmed case, outside China', color_discrete_sequence=[act, dth, rec])

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
group_data = data.groupby(['Date', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered'].sum()

group_data = group_data.reset_index()

group_data['Date'] = pd.to_datetime(group_data['Date'])

group_data['Date'] = group_data['Date'].dt.strftime('%m/%d/%Y')
group_data['size'] = group_data['Confirmed'].pow(0.3) # 47^0.3

fig = px.scatter_geo(group_data, locations="Country/Region", locationmode='country names', 

                     color="Confirmed", size='size', hover_name="Country/Region",

                     range_color= [0, max(group_data['Confirmed'])+2],

                     projection="equirectangular", animation_frame="Date", 

                     title='Confirmed Spread over time')

# fig.update(layout_coloraxis_showscale=False)

fig.show()
group_data['size'] = group_data['Deaths'].pow(0.3) # 47^0.3

fig = px.scatter_geo(group_data, locations="Country/Region", locationmode='country names', 

                     color="Deaths", size='size', hover_name="Country/Region",

                     range_color= [0, max(group_data['Deaths'])+2],

                     projection="equirectangular", animation_frame="Date", 

                     title='Deaths Spread over time')

# fig.update(layout_coloraxis_showscale=False)

fig.show()
group_data['size'] = group_data['Recovered'].pow(0.3) # 47^0.3

fig = px.scatter_geo(group_data, locations="Country/Region", locationmode='country names', 

                     color="Recovered", size='size', hover_name="Country/Region",

                     range_color= [0, max(group_data['Recovered'])+2],

                     projection="equirectangular", animation_frame="Date", 

                     title='Recovered Spread over time')

# fig.update(layout_coloraxis_showscale=False)

fig.show()
# temp = data.groupby(['Date', 'Country/Region'])['Confirmed', 'Active', 'Deaths', 'Recovered'].sum()

# temp = temp.reset_index().sort_values(by=['Date', 'Country/Region'])

# temp = temp[temp['Confirmed'] > 1000]



# plt.style.use('seaborn')

# g = sns.FacetGrid(temp, col="Country/Region", hue="Country/Region", 

#                   sharey=False, col_wrap=5)

# g = g.map(plt.plot, "Date", "Confirmed", color=cnf)

# g = g.map(plt.plot, "Date", "Active", color=act)

# g = g.map(plt.plot, "Date", "Deaths", color=dth)

# g = g.map(plt.plot, "Date", "Recovered", color=rec)

# g.set_xticklabels(rotation=90)

# plt.show()
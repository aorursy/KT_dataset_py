import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from itertools import cycle, islice
import seaborn as sb
import matplotlib.dates as dates
import datetime as dt

import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly import tools, subplots
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from pathlib import Path
import pandas as pd
pop_dir = Path('../input/countryinfo/covid19countryinfo.csv')
pop_data = pd.read_csv('../input/countryinfo/covid19countryinfo.csv')
pdata = pop_data[["country", "pop", "density"]]
pdata =pdata.rename(columns={'country':'Country_Region', 'pop':'Population'})

# convert columns population from String to float, to be able to divide
pdata['Population'] = pdata['Population'].str.replace(',', '')
pdata['Population'] = pdata['Population'].astype(float)

updata = pd.DataFrame(pdata.groupby('Country_Region')['Population'].max()).reset_index()
updata.head()
train_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")#index_col=0
display(train_data.head())
test_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")#index_col=0
display(test_data.head())

# train_data[train_data['Country_Region']== 'US']
sum_df = pd.pivot_table(train_data, values=['ConfirmedCases','Fatalities'], index=['Date'],aggfunc=np.sum)
display(sum_df.max())
train_data['NewConfirmedCases'] = train_data['ConfirmedCases'] - train_data['ConfirmedCases'].shift(1)
train_data['NewConfirmedCases'] = train_data['NewConfirmedCases'].fillna(0.0)
train_data['NewFatalities']     = train_data['Fatalities'] - train_data['Fatalities'].shift(1)
train_data['NewFatalities']     = train_data['NewFatalities'].fillna(0.0)#.astype(int)
train_data['MortalityRate']     = train_data['Fatalities'] / train_data['ConfirmedCases']
train_data['MortalityRate']     = train_data['MortalityRate'].fillna(0.0)
train_data['GrowthRate']        = train_data['NewConfirmedCases']/train_data['NewConfirmedCases'].shift(1)
train_data['GrowthRate']        = train_data['GrowthRate'].replace([-np.inf, np.inf],  0.0)
train_data['GrowthRate']        = train_data['GrowthRate'].fillna(0.0) 
display(train_data.head())
def getColumnInfo(df):
    n_province =  df['Province_State'].nunique()
    n_country  =  df['Country_Region'].nunique()
    n_days     =  df['Date'].nunique()
    start_date =  df['Date'].unique()[0]
    end_date   =  df['Date'].unique()[-1]
    return n_province, n_country, n_days, start_date, end_date

n_train = train_data.shape[0]
n_test = test_data.shape[0]

n_prov_train, n_count_train, n_train_days, start_date_train, end_date_train = getColumnInfo(train_data)
n_prov_test,  n_count_test,  n_test_days,  start_date_test,  end_date_test  = getColumnInfo(test_data)

print ('<==Train data==> \n # of Province_State: '+str(n_prov_train),', # of Country_Region:'+str(n_count_train), 
       ', Time Period: '+str(start_date_train)+' to '+str(end_date_train), '==> days:',str(n_train_days))
print("\n Countries with Province/State information:  ", train_data[train_data['Province_State'].isna()==False]['Country_Region'].unique())
print ('\n <==Test  data==> \n # of Province_State: '+str(n_prov_test),', # of Country_Region:'+str(n_count_test),
       ', Time Period: '+start_date_test+' to '+end_date_test, '==> days:',n_test_days)

df_test = test_data.loc[test_data.Date > '2020-04-14']
overlap_days = n_test_days - df_test.Date.nunique()
print('\n overlap days with training data: ', overlap_days, ', total days: ', n_train_days+n_test_days-overlap_days)
prob_confirm_check_train = train_data.ConfirmedCases.value_counts(normalize=True)
prob_fatal_check_train = train_data.Fatalities.value_counts(normalize=True)

n_confirm_train = train_data.ConfirmedCases.value_counts()[1:].sum()
n_fatal_train = train_data.Fatalities.value_counts()[1:].sum()

print('Percentage of confirmed case records = {0:<2.0f}/{1:<2.0f} = {2:<2.1f}%'.format(n_confirm_train, n_train, prob_confirm_check_train[1:].sum()*100))
print('Percentage of fatality records = {0:<2.0f}/{1:<2.0f} = {2:<2.1f}%'.format(n_fatal_train, n_train, prob_fatal_check_train[1:].sum()*100))
train_data_by_country = train_data.groupby(['Date','Country_Region'],as_index=False).agg({'ConfirmedCases': 'sum', 'Fatalities': 'sum',
                                                                                         'GrowthRate':'last' })

#display(train_data_by_country.tail(10))
max_train_date = train_data['Date'].max()
train_data_by_country_confirm = train_data_by_country.query('(Date == @max_train_date) & (ConfirmedCases > 100)').sort_values('ConfirmedCases', ascending=False)


#
train_data_by_country_confirm = train_data_by_country_confirm.merge(updata, on ="Country_Region", how='left')

train_data_by_country_confirm['InfectedPopulation_Ratio'] = train_data_by_country_confirm['ConfirmedCases'].div(train_data_by_country_confirm['Population'],fill_value=0)*100
train_data_by_country_confirm['InfectedPopulation_Ratio'] = train_data_by_country_confirm['InfectedPopulation_Ratio'].replace([-np.inf, np.inf],  0.0)


train_data_by_country_confirm.set_index('Country_Region', inplace=True)
train_data_by_country_confirm.style.background_gradient(cmap='Reds').format({'ConfirmedCases': "{:.0f}", 'GrowthRate': "{:.2f}", 'InfectedPopulation_Ratio': "{:.4f}"})

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_dark"
from plotly.subplots import make_subplots
import folium 
from folium import plugins
from tqdm.notebook import tqdm as tqdm

train_data_by_country_confirm= train_data_by_country_confirm.reset_index()
global_confirmedcases = train_data_by_country_confirm[['Country_Region','InfectedPopulation_Ratio']]

fig = px.bar(global_confirmedcases.sort_values('InfectedPopulation_Ratio',ascending=False)[:20],x='InfectedPopulation_Ratio',y='Country_Region',title='Country wise infected population ratio',text='InfectedPopulation_Ratio', height=900, orientation='h')
fig.show()

# without diamond princess
rest = global_confirmedcases[global_confirmedcases['Country_Region'] != 'Diamond Princess']
fig = px.bar(rest.sort_values('InfectedPopulation_Ratio',ascending=False)[:20],x='InfectedPopulation_Ratio',y='Country_Region',title='Country wise infected population ratio excluding Diamond Princess',text='InfectedPopulation_Ratio', height=900, orientation='h')
fig.show()
cleaned_data_mitigation = pd.read_csv('../input/covid19-containment-and-mitigation-measures/COVID 19 Containment measures data.csv', parse_dates=['Date Start'])
# cleaned_data_mitigation[cleaned_data_mitigation.Country == 'Spain']
grouped_china_mitigation = cleaned_data_mitigation[cleaned_data_mitigation['Country'] == "China"].reset_index()

def f(x):
     return pd.Series(dict(Keywords = "{%s}" % ', '.join(x['Keywords']), 
                        Description_of_measure_implemented = "{%s}" % ', '.join(x['Description of measure implemented'])))

grouped_china_mitigation_date = grouped_china_mitigation.groupby('Date Start').apply(f).reset_index()
grouped_china_mitigation_date.rename(columns={'Date Start':'Date'}, 
                 inplace=True)
print(grouped_china_mitigation_date)
cleaned_data = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', parse_dates=['Date'])
cleaned_data.head()
# cases 
cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']

# Active Case = confirmed - deaths - recovered
cleaned_data['Active'] = cleaned_data['Confirmed'] - cleaned_data['Deaths'] - cleaned_data['Recovered']

# filling missing values 
cleaned_data[['Province/State']] = cleaned_data[['Province/State']].fillna('')
cleaned_data[cases] = cleaned_data[cases].fillna(0)

cleaned_data.head()
grouped_china = cleaned_data[cleaned_data['Country/Region'] == "China"].reset_index()
grouped_china_date = grouped_china.groupby('Date')['Date', 'Confirmed', 'Deaths','Recovered','Active'].sum().reset_index()
print(grouped_china_date)
print(len(grouped_china_date.index))

grouped_italy = cleaned_data[cleaned_data['Country/Region'] == "Italy"].reset_index()
grouped_italy_date = grouped_italy.groupby('Date')['Date', 'Confirmed', 'Deaths','Recovered','Active'].sum().reset_index()

grouped_iran = cleaned_data[cleaned_data['Country/Region'] == "Iran"].reset_index()
grouped_iran_date = grouped_iran.groupby('Date')['Date', 'Confirmed', 'Deaths','Recovered','Active'].sum().reset_index()

grouped_korea = cleaned_data[cleaned_data['Country/Region'] == "South Korea"].reset_index()
grouped_korea_date = grouped_korea.groupby('Date')['Date', 'Confirmed', 'Deaths','Recovered','Active'].sum().reset_index()

grouped_spain = cleaned_data[cleaned_data['Country/Region'] == "Spain"].reset_index()
grouped_spain_date = grouped_spain.groupby('Date')['Date', 'Confirmed', 'Deaths','Recovered','Active'].sum().reset_index()

grouped_rest = cleaned_data[~cleaned_data['Country/Region'].isin(['China', 'Italy', 'iran', 'South Korea', 'Spain'])].reset_index()
grouped_rest_date = grouped_rest.groupby('Date')['Date', 'Confirmed', 'Deaths','Recovered','Active'].sum().reset_index()

keywords = []
measures = []
for date in grouped_china_date['Date']:
    keyword_value = ''
    measure_value = ''
    for dateStart in grouped_china_mitigation_date['Date']:
        if date == dateStart:
            keyword_value = grouped_china_mitigation_date[grouped_china_mitigation_date['Date'] == dateStart]['Keywords']
            measure_value = grouped_china_mitigation_date[grouped_china_mitigation_date['Date'] == dateStart]['Description_of_measure_implemented']
            break
    keywords.append(keyword_value)
    measures.append(measure_value)
grouped_china_date['Keywords'] = keywords
grouped_china_date['Description_of_measure_implemented'] = measures
print(grouped_china_date)

fig = make_subplots(rows=1, cols=1)

# trace1 = go.Scatter(x=grouped_china_date['Date'],y=grouped_china_date['Confirmed'],hovertext=keywords,
#                     name="Confirmed",line_color='yellow',mode='lines+markers',opacity=0.8, showlegend = False)
# trace2 = go.Scatter(x=grouped_china_date['Date'],y=grouped_china_date['Deaths'],hovertext=keywords,
#                     name="Deaths",line_color='red',mode='lines+markers',opacity=0.8, showlegend = False)
# trace3 = go.Scatter(x=grouped_china_date['Date'],y=grouped_china_date['Recovered'],hovertext=keywords,
#                     name="Recovered",line_color='green',mode='lines+markers',opacity=0.8, showlegend = False)
trace4 = go.Scatter(x=grouped_china_date['Date'],y=grouped_china_date['Active'],hovertext=keywords,
                    name="Active",line_color='orange',mode='lines+markers',opacity=0.8, showlegend = False)
# fig.append_trace(trace1, 1, 1)
# fig.append_trace(trace2, 2, 1)
# fig.append_trace(trace3, 3, 1)
fig.append_trace(trace4, 1, 1)

fig.update_layout(template="plotly_dark",title_text = '<b>Spread of the COVID19 over time in CHINA</b>',
                  hoverlabel_align = 'right',hovermode='closest', height=300#1200
                  , width=800, font=dict(family="Arial, Balto, Courier New, Droid Sans",color='white'))
fig.show()
fig = make_subplots(rows=1, cols=1)

# trace1 = go.Scatter(x=grouped_italy_date['Date'],y=grouped_italy_date['Confirmed'],hovertext=keywords,
#                     name="Confirmed",line_color='yellow',mode='lines+markers',opacity=0.8, showlegend = False)
# trace2 = go.Scatter(x=grouped_italy_date['Date'],y=grouped_italy_date['Deaths'],hovertext=keywords,
#                     name="Deaths",line_color='red',mode='lines+markers',opacity=0.8, showlegend = False)
# trace3 = go.Scatter(x=grouped_italy_date['Date'],y=grouped_italy_date['Recovered'],hovertext=keywords,
#                     name="Recovered",line_color='green',mode='lines+markers',opacity=0.8, showlegend = False)
trace4 = go.Scatter(x=grouped_italy_date['Date'],y=grouped_italy_date['Active'],hovertext=keywords,
                    name="Active",line_color='orange',mode='lines+markers',opacity=0.8, showlegend = False)
# fig.append_trace(trace1, 1, 1)
# fig.append_trace(trace2, 2, 1)
# fig.append_trace(trace3, 3, 1)
fig.append_trace(trace4, 1, 1)

fig.update_layout(template="plotly_dark",title_text = '<b>Spread of the COVID19 over time in ITALY</b>',
                  hoverlabel_align = 'right',hovermode='closest', height=300#1200
                  , width=800, font=dict(family="Arial, Balto, Courier New, Droid Sans",color='white'))
fig.show()
keywords = []
measures = []
for date in grouped_south_korea_date['Date']:
    keyword_value = ''
    measure_value = ''
    for dateStart in grouped_south_korea_mitigation_date['Date']:
        if date == dateStart:
            keyword_value = grouped_south_korea_mitigation_date[grouped_south_korea_mitigation_date['Date'] == dateStart]['Keywords']
            measure_value = grouped_south_korea_mitigation_date[grouped_south_korea_mitigation_date['Date'] == dateStart]['Description_of_measure_implemented']
            break
    keywords.append(keyword_value)
    measures.append(measure_value)
grouped_south_korea_date['Keywords'] = keywords
grouped_south_korea_date['Description_of_measure_implemented'] = measures
fig = make_subplots(rows=1, cols=1)

# trace1 = go.Scatter(x=grouped_south_korea_date['Date'],y=grouped_south_korea_date['Confirmed'],hovertext=keywords,
#                     name="Confirmed",line_color='yellow',mode='lines+markers',opacity=0.8, showlegend = False)
# trace2 = go.Scatter(x=grouped_south_korea_date['Date'],y=grouped_south_korea_date['Deaths'],hovertext=keywords,
#                     name="Deaths",line_color='red',mode='lines+markers',opacity=0.8, showlegend = False)
# trace3 = go.Scatter(x=grouped_south_korea_date['Date'],y=grouped_south_korea_date['Recovered'],hovertext=keywords,
#                     name="Recovered",line_color='green',mode='lines+markers',opacity=0.8, showlegend = False)
trace4 = go.Scatter(x=grouped_south_korea_date['Date'],y=grouped_south_korea_date['Active'],hovertext=keywords,
                    name="Active",line_color='orange',mode='lines+markers',opacity=0.8, showlegend = False)
# fig.append_trace(trace1, 1, 1)
# fig.append_trace(trace2, 2, 1)
# fig.append_trace(trace3, 3, 1)
fig.append_trace(trace4, 1, 1)

fig.update_layout(template="plotly_dark",title_text = '<b>Spread of the COVID19 over time in SOUTH KOREA</b>',
                  hoverlabel_align = 'right',hovermode='closest', height=300#1200
                  , width=800, font=dict(family="Arial, Balto, Courier New, Droid Sans",color='white'))
fig.show()
keywords = []
measures = []
for date in grouped_india_date['Date']:
    keyword_value = ''
    measure_value = ''
    for dateStart in grouped_india_mitigation_date['Date']:
        if date == dateStart:
            keyword_value = grouped_india_mitigation_date[grouped_india_mitigation_date['Date'] == dateStart]['Keywords']
            measure_value = grouped_india_mitigation_date[grouped_india_mitigation_date['Date'] == dateStart]['Description_of_measure_implemented']
            break
    keywords.append(keyword_value)
    measures.append(measure_value)
grouped_india_date['Keywords'] = keywords
grouped_india_date['Description_of_measure_implemented'] = measures


fig = make_subplots(rows=1, cols=1)

# trace1 = go.Scatter(x=grouped_india_date['Date'],y=grouped_india_date['Confirmed'],hovertext=keywords,
#                     name="Confirmed",line_color='yellow',mode='lines+markers',opacity=0.8, showlegend = False)
# trace2 = go.Scatter(x=grouped_india_date['Date'],y=grouped_india_date['Deaths'],hovertext=keywords,
#                     name="Deaths",line_color='red',mode='lines+markers',opacity=0.8, showlegend = False)
# trace3 = go.Scatter(x=grouped_india_date['Date'],y=grouped_india_date['Recovered'],hovertext=keywords,
#                     name="Recovered",line_color='green',mode='lines+markers',opacity=0.8, showlegend = False)
trace4 = go.Scatter(x=grouped_india_date['Date'],y=grouped_india_date['Active'],hovertext=keywords,
                    name="Active",line_color='orange',mode='lines+markers',opacity=0.8, showlegend = False)
# fig.append_trace(trace1, 1, 1)
# fig.append_trace(trace2, 2, 1)
# fig.append_trace(trace3, 3, 1)
fig.append_trace(trace4, 1, 1)

fig.update_layout(template="plotly_dark",title_text = '<b>Spread of the COVID19 over time in INDIA</b>',
                  hoverlabel_align = 'right',hovermode='closest', height=300#1200
                  , width=800, font=dict(family="Arial, Balto, Courier New, Droid Sans",color='white'))
fig.show()
keywords = []
measures = []
for date in grouped_us_date['Date']:
    keyword_value = ''
    measure_value = ''
    for dateStart in grouped_us_mitigation_date['Date']:
        if date == dateStart:
            keyword_value = grouped_us_mitigation_date[grouped_us_mitigation_date['Date'] == dateStart]['Keywords']
            measure_value = grouped_us_mitigation_date[grouped_us_mitigation_date['Date'] == dateStart]['Description_of_measure_implemented']
            break
    keywords.append(keyword_value)
    measures.append(measure_value)
grouped_us_date['Keywords'] = keywords
grouped_us_date['Description_of_measure_implemented'] = measures


fig = make_subplots(rows=1, cols=1)

# trace1 = go.Scatter(x=grouped_us_date['Date'],y=grouped_us_date['Confirmed'],hovertext=keywords,
#                     name="Confirmed",line_color='yellow',mode='lines+markers',opacity=0.8, showlegend = False)
# trace2 = go.Scatter(x=grouped_us_date['Date'],y=grouped_us_date['Deaths'],hovertext=keywords,
#                     name="Deaths",line_color='red',mode='lines+markers',opacity=0.8, showlegend = False)
# trace3 = go.Scatter(x=grouped_us_date['Date'],y=grouped_us_date['Recovered'],hovertext=keywords,
#                     name="Recovered",line_color='green',mode='lines+markers',opacity=0.8, showlegend = False)
trace4 = go.Scatter(x=grouped_us_date['Date'],y=grouped_us_date['Active'],hovertext=keywords,
                    name="Active",line_color='orange',mode='lines+markers',opacity=0.8, showlegend = False)
# fig.append_trace(trace1, 1, 1)
# fig.append_trace(trace2, 2, 1)
# fig.append_trace(trace3, 3, 1)
fig.append_trace(trace4, 1, 1)

fig.update_layout(template="plotly_dark",title_text = '<b>Spread of the COVID19 over time in US</b>',
                  hoverlabel_align = 'right',hovermode='closest', height=300#1200
                  , width=800, font=dict(family="Arial, Balto, Courier New, Droid Sans",color='white'))
fig.show()
keywords = []
measures = []
for date in grouped_singapore_date['Date']:
    keyword_value = ''
    measure_value = ''
    for dateStart in grouped_singapore_mitigation_date['Date']:
        if date == dateStart:
            keyword_value = grouped_singapore_mitigation_date[grouped_singapore_mitigation_date['Date'] == dateStart]['Keywords']
            measure_value = grouped_singapore_mitigation_date[grouped_singapore_mitigation_date['Date'] == dateStart]['Description_of_measure_implemented']
            break
    keywords.append(keyword_value)
    measures.append(measure_value)
grouped_singapore_date['Keywords'] = keywords
grouped_singapore_date['Description_of_measure_implemented'] = measures


fig = make_subplots(rows=1, cols=1)

# trace1 = go.Scatter(x=grouped_singapore_date['Date'],y=grouped_singapore_date['Confirmed'],hovertext=keywords,
#                     name="Confirmed",line_color='yellow',mode='lines+markers',opacity=0.8, showlegend = False)
# trace2 = go.Scatter(x=grouped_singapore_date['Date'],y=grouped_singapore_date['Deaths'],hovertext=keywords,
#                     name="Deaths",line_color='red',mode='lines+markers',opacity=0.8, showlegend = False)
# trace3 = go.Scatter(x=grouped_singapore_date['Date'],y=grouped_singapore_date['Recovered'],hovertext=keywords,
#                     name="Recovered",line_color='green',mode='lines+markers',opacity=0.8, showlegend = False)
trace4 = go.Scatter(x=grouped_singapore_date['Date'],y=grouped_singapore_date['Active'],hovertext=keywords,
                    name="Active",line_color='orange',mode='lines+markers',opacity=0.8, showlegend = False)
# fig.append_trace(trace1, 1, 1)
# fig.append_trace(trace2, 2, 1)
# fig.append_trace(trace3, 3, 1)
fig.append_trace(trace4, 1, 1)

fig.update_layout(template="plotly_dark",title_text = '<b>Spread of the COVID19 over time in SINGAPORE</b>',
                  hoverlabel_align = 'right',hovermode='closest'#,height=300#1200
                  , width=800, font=dict(family="Arial, Balto, Courier New, Droid Sans",color='white'))
fig.show()
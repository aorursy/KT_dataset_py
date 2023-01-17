import numpy as np
import pandas as pd
!pip install quandl
import quandl

# It is recommended to use your API Key.
# You can get your API key from, https://www.quandl.com/data/BSE-Bombay-Stock-Exchange/usage/quickstart/api
# quandl.ApiConfig.api_key = ""
sector_indices = ['BSE/SI1900', 'BSE/SIBANK', 'BSE/SI0400', 'BSE/SPBSFIIP', 'BSE/SI0600', 
                  'BSE/SI1000', 'BSE/SI0800', 'BSE/SPBSBMIP', 'BSE/SI0200', 'BSE/CARBON', 
                  'BSE/SPBSCDIP', 'BSE/SPBSENIP', 'BSE/GREENX' ,'BSE/SPBSIDIP', 'BSE/SI1200', 
                  'BSE/SI1400', 'BSE/SIPOWE', 'BSE/SIBPSU', 'BSE/SIREAL', 'BSE/SIBTEC', 
                  'BSE/SPBSTLIP', 'BSE/SPBSUTIP', 'BSE/INFRA', 'BSE/SPBIMIP']

sector_name = ['AUTO', 'BANK', 'CONSUMER_DURABLES', 'FINANCE', 'FMCG', 'IT', 'HEALTHCARE', 
               'BASIC_MATERIAL', 'CAP_GOODS', 'CARBONEX', 'CONSUMER_DISCRETIONARY_GOODS', 
               'ENERGY', 'GREENEX', 'INDUSTRIALS', 'METAL', 'OIL_GAS', 'POWER', 'PSU', 'REALTY', 
               'TECK', 'TELECOM', 'UTILITIES', 'INFRASTRUCTURE', 'MANUFACTURING']

print('Total Sector considered: ', len(sector_name))
from datetime import datetime
import pandas as pd
import numpy as np
start = '2019-12-01'
end = datetime.today().strftime('%Y-%m-%d')

bse_sector_indices = pd.DataFrame()
for i in range(0, len(sector_indices)):
    print("Sector: "+sector_name[i])
    bse_sector_indices[sector_name[i]] = quandl.get(
        sector_indices[i], start_date=start, 
        end_date=end)['Close'].pct_change()
    bse_sector_indices[sector_name[i]] = np.round((bse_sector_indices[sector_name[i]]), 3)
bse_sector_indices = bse_sector_indices.dropna()
bse_sector_indices = bse_sector_indices.reset_index()
global_ = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
india = global_[global_['Country/Region']=='India'].drop(['Lat','Long','Province/State'], 1)
india = pd.melt(india, id_vars='Country/Region')
india = india.rename({'variable':'Date','value':'Cases'}, axis=1)
india['Date'] = pd.to_datetime(india['Date'], format='%m/%d/%y')
india.Date = india.Date.dt.strftime('%Y-%m-%d')
india['Date'] = pd.to_datetime(india['Date'], format='%Y-%m-%d')
india.tail()
before_corona = pd.DataFrame()
before_corona['Country/Region'] = "India"
before_corona['Date'] = pd.date_range(start ='2019-12-01',
                                      end ='2020-01-21', 
                                      freq ='D')
before_corona['Cases'] = 0
india = before_corona.append(india)
india.head()
finance_analysis = pd.merge(bse_sector_indices, india[['Date','Cases']], on='Date')
finance_analysis.head()
finance_analysis['Month'] = finance_analysis.Date.dt.month
finance_analysis['Day'] = finance_analysis.Date.dt.day
finance_analysis['Weekday'] = finance_analysis.Date.dt.dayofweek
finance_analysis.tail()
before_corona = finance_analysis[(finance_analysis['Date']>='2019-01-01') & (finance_analysis['Date']<='2020-01-29')]
before_lockdown = finance_analysis[(finance_analysis['Date']>='2020-01-30') & (finance_analysis['Date']<='2020-03-24')]
phase_1 = finance_analysis[(finance_analysis['Date']>='2020-03-25') & (finance_analysis['Date']<='2020-04-14')]
phase_2 = finance_analysis[(finance_analysis['Date']>='2020-04-15') & (finance_analysis['Date']<='2020-05-03')]
phase_3 = finance_analysis[(finance_analysis['Date']>='2020-05-04') & (finance_analysis['Date']<='2020-05-17')]
phase_4 = finance_analysis[(finance_analysis['Date']>='2020-05-18') & (finance_analysis['Date']<='2020-05-31')]
phase_5 = finance_analysis[(finance_analysis['Date']>='2020-06-01') & (finance_analysis['Date']<='2020-06-30')]
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
%matplotlib inline
avg_return = pd.DataFrame()
avg_return['Sector'] = sector_name
avg_return['Avg_Return'] = np.round(before_lockdown.mean().head(len(sector_name)).values, 3)

fig = px.bar(avg_return, x='Sector', y='Avg_Return',
             hover_name='Avg_Return', color='Avg_Return', text='Avg_Return',
             height=600, title='Average Returns of each sector before lockdown')
fig.update_traces(texttemplate='%{text:.s}', textposition='outside')
fig.show()
correlation_stock = before_lockdown.corr()['Cases'].head(-4)
correlation_stock = np.round(correlation_stock, 3)
fig = px.bar(correlation_stock, x=correlation_stock.index, y='Cases',
             hover_name='Cases', color='Cases', text='Cases',
             height=600, title='Correlation Graph with each sector returns and COVID-19 Cases before lockdown')
fig.update_traces(texttemplate='%{text:.s}', textposition='outside')
fig.show()
avg_return = pd.DataFrame()
avg_return['Sector'] = sector_name
avg_return['Avg_Return'] = np.round(phase_1.mean().head(len(sector_name)).values, 3)

fig = px.bar(avg_return, x='Sector', y='Avg_Return',
             hover_name='Avg_Return', color='Avg_Return', text='Avg_Return',
             height=600, title='Average Returns of each sector on Phase - 1 of lockdown')
fig.update_traces(texttemplate='%{text:.s}', textposition='outside')
fig.show()
correlation_stock = phase_1.corr()['Cases'].head(-4)
correlation_stock = np.round(correlation_stock, 3)
fig = px.bar(correlation_stock, x=correlation_stock.index, y='Cases',
             hover_name='Cases', color='Cases', text='Cases',
             height=600, 
             title='Correlation Graph with each sector returns and COVID-19 Cases on Phase-1 of lockdown')
fig.update_traces(texttemplate='%{text:.s}', textposition='outside')
fig.show()
avg_return = pd.DataFrame()
avg_return['Sector'] = sector_name
avg_return['Avg_Return'] = np.round(phase_2.mean().head(len(sector_name)).values, 3)

fig = px.bar(avg_return, x='Sector', y='Avg_Return',
             hover_name='Avg_Return', color='Avg_Return', text='Avg_Return',
             height=600, title='Average Returns of each sector on Phase - 2 of lockdown')
fig.update_traces(texttemplate='%{text:.s}', textposition='outside')
fig.show()
correlation_stock = phase_2.corr()['Cases'].head(-4)
correlation_stock = np.round(correlation_stock, 3)
fig = px.bar(correlation_stock, x=correlation_stock.index, y='Cases',
             hover_name='Cases', color='Cases', text='Cases',
             height=600, 
             title='Correlation Graph with each sector returns and COVID-19 Cases on Phase-2 of lockdown')
fig.update_traces(texttemplate='%{text:.s}', textposition='outside')
fig.show()
avg_return = pd.DataFrame()
avg_return['Sector'] = sector_name
avg_return['Avg_Return'] = np.round(phase_3.mean().head(len(sector_name)).values, 3)

fig = px.bar(avg_return, x='Sector', y='Avg_Return',
             hover_name='Avg_Return', color='Avg_Return', text='Avg_Return',
             height=600, title='Average Returns of each sector on Phase - 3 of lockdown')
fig.update_traces(texttemplate='%{text:.s}', textposition='outside')
fig.show()
correlation_stock = phase_3.corr()['Cases'].head(-4)
correlation_stock = np.round(correlation_stock, 3)
fig = px.bar(correlation_stock, x=correlation_stock.index, y='Cases',
             hover_name='Cases', color='Cases', text='Cases',
             height=600, 
             title='Correlation Graph with each sector returns and COVID-19 Cases on Phase-3 of lockdown')
fig.update_traces(texttemplate='%{text:.s}', textposition='outside')
fig.show()
avg_return = pd.DataFrame()
avg_return['Sector'] = sector_name
avg_return['Avg_Return'] = np.round(phase_4.mean().head(len(sector_name)).values, 3)

fig = px.bar(avg_return, x='Sector', y='Avg_Return',
             hover_name='Avg_Return', color='Avg_Return', text='Avg_Return',
             height=600, title='Average Returns of each sector on Phase - 4 of lockdown')
fig.update_traces(texttemplate='%{text:.s}', textposition='outside')
fig.show()
correlation_stock = phase_4.corr()['Cases'].head(-4)
correlation_stock = np.round(correlation_stock, 3)
fig = px.bar(correlation_stock, x=correlation_stock.index, y='Cases',
             hover_name='Cases', color='Cases', text='Cases',
             height=600, 
             title='Correlation Graph with each sector returns and COVID-19 Cases Phase-4 of lockdown')
fig.update_traces(texttemplate='%{text:.s}', textposition='outside')
fig.show()
avg_return = pd.DataFrame()
avg_return['Sector'] = sector_name
avg_return['Avg_Return'] = np.round(phase_5.mean().head(len(sector_name)).values, 3)

fig = px.bar(avg_return, x='Sector', y='Avg_Return',
             hover_name='Avg_Return', color='Avg_Return', text='Avg_Return',
             height=600, title='Average Returns of each sector on Phase - 5 of Unlock 1.0')
fig.update_traces(texttemplate='%{text:.s}', textposition='outside')
fig.show()
correlation_stock = phase_5.corr()['Cases'].head(-4)
correlation_stock = np.round(correlation_stock, 3)
fig = px.bar(correlation_stock, x=correlation_stock.index, y='Cases',
             hover_name='Cases', color='Cases', text='Cases',
             height=600, 
             title='Correlation Graph with each sector returns and COVID-19 Cases on Phase-5 of Unlock 1.0')
fig.update_traces(texttemplate='%{text:.s}', textposition='outside')
fig.show()
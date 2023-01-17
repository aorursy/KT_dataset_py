# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

from plotly.offline import init_notebook_mode, iplot

pio.templates.default = "plotly_dark"

from plotly.subplots import make_subplots
data = pd.read_csv('../input/covid19-corona-virus-india-dataset/complete.csv', parse_dates=['Date'])
data.rename(columns={'Name of State / UT': 'State', 

                     'Total Confirmed cases':'Confirmed',

                     'Death':'Deaths'}, inplace=True)



data['Mortality Rate'] = (100.* (data['Deaths']/data['Confirmed'])).round(1)
grouped_maharashtra = data[data['State'] == "Maharashtra"].reset_index()

grouped_maharashtra_date = grouped_maharashtra.groupby('Date')['Date', 'Confirmed', 'Deaths'].sum().reset_index()

grouped_maharashtra_date = grouped_maharashtra_date[grouped_maharashtra_date ['Confirmed']>= 100]

grouped_maharashtra_date['Day']=np.arange(len(grouped_maharashtra_date))



grouped_kerela = data[data['State'] == "Kerala"].reset_index()

grouped_kerela_date = grouped_kerela.groupby('Date')['Date', 'Confirmed', 'Deaths'].sum().reset_index()

grouped_kerela_date = grouped_kerela_date[grouped_kerela_date ['Confirmed']>= 100]

grouped_kerela_date['Day']=np.arange(len(grouped_kerela_date))



grouped_gujarat = data[data['State'] == "Gujarat"].reset_index()

grouped_gujarat_date = grouped_gujarat.groupby('Date')['Date', 'Confirmed', 'Deaths'].sum().reset_index()

grouped_gujarat_date = grouped_gujarat_date[grouped_gujarat_date ['Confirmed']>= 100]

grouped_gujarat_date['Day']=np.arange(len(grouped_gujarat_date))



grouped_tn = data[data['State'] == "Tamil Nadu"].reset_index()

grouped_tn_date = grouped_tn.groupby('Date')['Date', 'Confirmed', 'Deaths'].sum().reset_index()

grouped_tn_date = grouped_tn_date[grouped_tn_date ['Confirmed']>= 100]

grouped_tn_date['Day']=np.arange(len(grouped_tn_date))



grouped_up = data[data['State'] == "Uttar Pradesh"].reset_index()

grouped_up_date = grouped_up.groupby('Date')['Date', 'Confirmed', 'Deaths'].sum().reset_index()

grouped_up_date = grouped_up_date[grouped_up_date ['Confirmed']>= 100]

grouped_up_date['Day']=np.arange(len(grouped_up_date))



grouped_delhi = data[data['State'] == "Delhi"].reset_index()

grouped_delhi_date = grouped_delhi.groupby('Date')['Date', 'Confirmed', 'Deaths'].sum().reset_index()

grouped_delhi_date = grouped_delhi_date[grouped_delhi_date ['Confirmed']>= 100]

grouped_delhi_date['Day']=np.arange(len(grouped_delhi_date))



grouped_karnataka = data[data['State'] == "Karnataka"].reset_index()

grouped_karnataka_date = grouped_karnataka.groupby('Date')['Date', 'Confirmed', 'Deaths'].sum().reset_index()

grouped_karnataka_date = grouped_karnataka_date[grouped_karnataka_date ['Confirmed']>= 100]

grouped_karnataka_date['Day']=np.arange(len(grouped_karnataka_date))



grouped_wb = data[data['State'] == "West Bengal"].reset_index()

grouped_wb_date = grouped_wb.groupby('Date')['Date', 'Confirmed', 'Deaths'].sum().reset_index()

grouped_wb_date = grouped_wb_date[grouped_wb_date ['Confirmed']>= 100]

grouped_wb_date['Day']=np.arange(len(grouped_wb_date))



grouped_bihar = data[data['State'] == "Bihar"].reset_index()

grouped_bihar_date = grouped_bihar.groupby('Date')['Date', 'Confirmed', 'Deaths'].sum().reset_index()

grouped_bihar_date = grouped_bihar_date[grouped_bihar_date ['Confirmed']>= 100]

grouped_bihar_date['Day']=np.arange(len(grouped_bihar_date))
print("Data Information")

print(f"Earliest Entry: {data['Date'].min()}")

print(f"Last Entry:     {data['Date'].max()}")

print(f"Total Days:     {data['Date'].max() - data['Date'].min()}")
from datetime import date, timedelta



state=[]

growth_rate=[]



recent_date= grouped_karnataka_date['Date'].max()

week_old = recent_date - timedelta(7)

x = grouped_karnataka_date[grouped_karnataka_date['Date']== recent_date]['Confirmed'].values[0]

y = grouped_karnataka_date[grouped_karnataka_date['Date']== week_old]['Confirmed'].values[0]

state.append('Karnataka')

growth_rate.append((x/y)**(1/7)-1)



recent_date= grouped_up_date['Date'].max()

week_old = recent_date - timedelta(7)

x = grouped_up_date[grouped_up_date['Date']== recent_date]['Confirmed'].values[0]

y = grouped_up_date[grouped_up_date['Date']== week_old]['Confirmed'].values[0]

state.append('Uttar Pradesh')

growth_rate.append((x/y)**(1/7)-1)



recent_date= grouped_kerela_date['Date'].max()

week_old = recent_date - timedelta(7)

x = grouped_kerela_date[grouped_kerela_date['Date']== recent_date]['Confirmed'].values[0]

y = grouped_kerela_date[grouped_kerela_date['Date']== week_old]['Confirmed'].values[0]

state.append('Kerala')

growth_rate.append((x/y)**(1/7)-1)



recent_date= grouped_maharashtra_date['Date'].max()

week_old = recent_date - timedelta(7)

x = grouped_maharashtra_date[grouped_maharashtra_date['Date']== recent_date]['Confirmed'].values[0]

y = grouped_maharashtra_date[grouped_maharashtra_date['Date']== week_old]['Confirmed'].values[0]

state.append('Maharashtra')

growth_rate.append((x/y)**(1/7)-1)



recent_date= grouped_tn_date['Date'].max()

week_old = recent_date - timedelta(7)

x = grouped_tn_date[grouped_tn_date['Date']== recent_date]['Confirmed'].values[0]

y = grouped_tn_date[grouped_tn_date['Date']== week_old]['Confirmed'].values[0]

state.append('Tamil Nadu')

growth_rate.append((x/y)**(1/7)-1)



recent_date= grouped_gujarat_date['Date'].max()

week_old = recent_date - timedelta(7)

x = grouped_gujarat_date[grouped_gujarat_date['Date']== recent_date]['Confirmed'].values[0]

y = grouped_gujarat_date[grouped_gujarat_date['Date']== week_old]['Confirmed'].values[0]

state.append('Gujarat')

growth_rate.append((x/y)**(1/7)-1)



recent_date= grouped_delhi_date['Date'].max()

week_old = recent_date - timedelta(7)

x = grouped_delhi_date[grouped_delhi_date['Date']== recent_date]['Confirmed'].values[0]

y = grouped_delhi_date[grouped_delhi_date['Date']== week_old]['Confirmed'].values[0]

state.append('Delhi')

growth_rate.append((x/y)**(1/7)-1)



recent_date= grouped_wb_date['Date'].max()

week_old = recent_date - timedelta(7)

x = grouped_wb_date[grouped_wb_date['Date']== recent_date]['Confirmed'].values[0]

y = grouped_wb_date[grouped_wb_date['Date']== week_old]['Confirmed'].values[0]

state.append('West Bengal')

growth_rate.append((x/y)**(1/7)-1)



recent_date= grouped_bihar_date['Date'].max()

week_old = recent_date - timedelta(7)

x = grouped_bihar_date[grouped_bihar_date['Date']== recent_date]['Confirmed'].values[0]

y = grouped_bihar_date[grouped_bihar_date['Date']== week_old]['Confirmed'].values[0]

state.append('Bihar')

growth_rate.append((x/y)**(1/7)-1)
dict = {'State':state,'Growth Rate':growth_rate}

growth_rate = pd.DataFrame(dict)
data[data['Date']==data['Date'].max()].sort_values(by=['Confirmed'],ascending= False, inplace=True)

data
growth_rate.sort_values(by='Growth Rate',ascending=False,inplace=True)

growth_rate
fig = go.Figure()



fig.add_trace(go.Scatter(x= grouped_kerela_date.Day, y= grouped_kerela_date.Confirmed, name = 'Kerela',

                         line=dict(color='firebrick', width=4)))



fig.add_trace(go.Scatter(x= grouped_maharashtra_date.Day, y= grouped_maharashtra_date.Confirmed, name ='Maharashtra',

                         line=dict(color='royalblue', width=4)))



fig.add_trace(go.Scatter(x= grouped_tn_date.Day, y= grouped_tn_date.Confirmed, name='Tamil Nadu',

                         line=dict(color='yellow', width=4)))



fig.add_trace(go.Scatter(x= grouped_up_date.Day, y= grouped_up_date.Confirmed, name='Uttar Pradesh',

                         line=dict(color='goldenrod', width=4)))



fig.add_trace(go.Scatter(x= grouped_karnataka_date.Day, y= grouped_karnataka_date.Confirmed, name='Karnataka',

                         line=dict(color='pink', width=4)))



fig.add_trace(go.Scatter(x= grouped_delhi_date.Day, y= grouped_delhi_date.Confirmed, name='Delhi',

                         line=dict(color='green', width=4)))



fig.add_trace(go.Scatter(x= grouped_gujarat_date.Day, y= grouped_gujarat_date.Confirmed, name='Gujarat',

                         line=dict(color='orange', width=4)))



fig.add_trace(go.Scatter(x= grouped_wb_date.Day, y= grouped_wb_date.Confirmed, name='West Bengal',

                         line=dict(color='purple', width=4)))



fig.update_layout(title='Curve comparing COVID-19 in different States of India',

                   xaxis_title='Day',

                   yaxis_title='Confirmed Cases')



#fig.update_yaxes(type="log")



fig.show()
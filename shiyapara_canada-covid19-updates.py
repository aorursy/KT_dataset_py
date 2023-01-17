from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

%matplotlib inline

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from matplotlib import ticker 

import seaborn as sns

from datetime import datetime, timedelta,date

import plotly.express as px





#from plotly import __version__

#import cufflinks as cf

import plotly.offline as py

import plotly.express as px

#py.initnotebookmode(connected=True)

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as py

init_notebook_mode(connected=True)

init_notebook_mode(connected=True)

import plotly.tools as tls

import plotly.graph_objs as gobj

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

nRowsRead = 1000 # specify 'None' if want to read whole file

# upload data files: covid19 & province_code

df = pd.read_csv('/kaggle/input/covid19_2404.csv',parse_dates = ['date'])

df.dataframeName = 'covid19_2404.csv'

df1 = pd.read_csv('/kaggle/input/P_code.csv')

df1.dataframeName = 'P_code.csv'

events= pd.read_csv('/kaggle/input/Canada_events.csv')

events.dataframeName = 'Canada_events.csv'

df2 = pd.merge(df,

                df1,

                on = 'prname',

                how = 'inner')

#pd.to_datetime(df2['date'], infer_datetime_format =True)

#df2 = df2.sort_values(by="date")

df2.tail(2)
# change object to number

df2['Population']=pd.to_numeric(df2['Population'],errors ='coerce')
df2.info()
# Data cleaning and manipulation

# Drop columns

df2.drop(['pruid','prnameFR','numprob','percentrecover','ratetested','percentoday'],axis =1, inplace =True)

df2.head(2)
# Replace following column names with: 

#'prname' with 'Province' |'date' with 'Date'|'numconf' with 'Confirmed'|'numdeaths' with 'Deaths'

# 'numtotal' with 'Total' |  'numtested' with 'Tested' |'numrecover' with 'Recovered'

df2.columns = ['Province','Date','Confirmed','Deaths','Total','Tested','Recovered', 'Daily_cases', 'P_code','Population']

df2.head(2)
# Drop "Repatriated travellers" from Province

df3 = df2.set_index('P_code')

df3.drop(['RT','CA'], inplace = True)

df3['Tested']=df3['Tested'].fillna('0')

df3['Recovered']=df3['Recovered'].fillna('0')

df3['Daily_cases']=df3['Daily_cases'].fillna('0')

df3.head(4)
fig1 = px.line(df3, x='Confirmed', y='Deaths',color ='Province')

fig1.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',

title="Fig1:Trajectory of COVID-19 in Canada",              

        xaxis_title =" # of Confirmed Cases",

        #xaxis_type ="log",

        yaxis_title = "# of Deaths (log scale)",

        yaxis_type ="log")

py.offline.iplot(fig1)
fig2 = px.line(df3, x='Date', y='Confirmed', color ='Province')

fig2.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',

title="Fig2: Trajectory of COVID-19 Confirmed Cases - Canada",              

        xaxis_title =" Date",

        yaxis_title = "# of Confirmed Cases")

        #yaxis_type ="log")

py.offline.iplot(fig2)
fig3 = px.line(df3, x='Date', y='Deaths',color ='Province')

fig3.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',

title="Fig3: Trajectory of COVID-19 Deaths - Canada",              

        xaxis_title =" # of Confirmed Cases",

        yaxis_title = "# of Deaths",

        yaxis_type ="log")

py.offline.iplot(fig3)
events['Death'] = events['Death'].fillna('0')

pd.to_datetime(events['Date'], infer_datetime_format =True)

events.drop('Description', axis = 1, inplace = True)

events.head()
# plotted the timeline X = date vs Y = confirmed cases based on at the day of event.



fig = px.scatter(events, x='Date', y='Confirmed', size_max = 14, hover_name = 'CA_events', color ='CA_events')

fig.update_traces(marker=dict(size=8,line=dict(width =2, color ='DarkSlateGrey')),

                 selector=dict(mode='markers')),

fig.update_layout(uniformtext_minsize=5,uniformtext_mode='hide',

        title="Timeline of COVID-19 Confirmed Cases in Canada", 

        width =1200,

        height=700,

        xaxis_title =" Date (since January 31, 2020)",

        yaxis_title = "# of Confirmed Cases"),

        #xaxis.set_major_formatter(mdates.DateFormatter('%b %d')

py.offline.iplot(fig)
# Correlation test is performed to understand the relationship between confirmed, deaths and total cases

df3.corr()

plt.figure(figsize=(6,4))

sns.heatmap(df3.corr(), annot =True)
#df4 = df2.set_index('P_code')

df3['Tested']=df3['Tested'].fillna('0')

df3['Recovered']=df3['Recovered'].fillna('0')

df3['Daily_cases']=df3['Daily_cases'].fillna('0')

df3.tail(2)


m = df3[df3['Date']== '24-04-2020']

m =df3.groupby('Date')['Confirmed', 'Deaths', 'Tested','Total','Recovered', 'Daily_cases'].sum()

m = df3.sort_values('Confirmed', ascending =  False).reset_index()

m ['mortality_rate'] = (m['Deaths'] / m ['Confirmed'])*100.0

m ['mortality_rate'] = m ['mortality_rate'].fillna('0')

m ['morbidity_rate'] = (m['Confirmed'] / m ['Population'])*100.0

m ['morbidity_rate'] = m ['morbidity_rate'].fillna('0')

m.head(5)
fig4 =px.area(m, x = 'Date', y = 'mortality_rate', color = 'Province')

fig4.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',

title=" Fig:4: COVID-19 Mortality Rates by Province",              

        xaxis_title =" Dates",

        yaxis_title = "Mortality Rate (%)")

        #yaxis_type ="log")

py.offline.iplot(fig4)
fig5 =px.area(m, x = 'Date', y = 'morbidity_rate', color = 'Province')

fig5.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',

title=" Fig:5: COVID-19 Morbidity Rates by Province",              

        xaxis_title =" Dates",

        yaxis_title = "Morbidity Rate (%)")

        #yaxis_type ="log"

py.offline.iplot(fig5)
new_cases = df2.set_index('P_code')

#new_cases = df2[df2['Date']== '24-04-2020']

#new_cases =df2.groupby('Date')['Confirmed', 'Deaths', 'Tested','Total','Recovered', 'Daily_cases'].sum()

new_cases.drop (['RT',"AB",'BC','MB',"SK",'NB','NL','NS','NWT','YT','NU','PE'], inplace =True)

new_cases.head()
fig6 =px.area(new_cases, x = 'Date', y = 'Daily_cases', color = 'Province')

fig6.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',

title=" Fig:6: New Cases by Date",              

        xaxis_title =" Dates",

        yaxis_title = "New Cases(#)")

        #yaxis_type ="log"

py.offline.iplot(fig6)
density = df2.set_index('P_code')

density.drop (['RT',"AB",'BC','MB',"SK",'NB','NL','NS','NWT','YT','NU','PE','ON','QC'], inplace =True)

density.head()
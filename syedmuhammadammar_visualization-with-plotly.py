import pandas as pd
import numpy as np
from plotly.offline import iplot
import plotly as py
import plotly.tools as tls
import matplotlib
from matplotlib import pyplot as plt
import cufflinks as cf 
%matplotlib inline
py.offline.init_notebook_mode(connected=True)
cf.go_offline()
# from google.colab import files
# uploaded = files.upload()




import io
# df_original = pd.read_csv(io.BytesIO(uploaded['train.csv']))
df_original = pd.read_csv('../input/countries-covid19jan-to-apr/train.csv')
df_original['Date'] = pd.to_datetime(df_original['Date'])
df_original
# import plotly.io as pio
# pio.renderers.default = 'notebook'
    
import plotly.graph_objects as go
world_graph_1 = df_original[['ConfirmedCases','Fatalities']]
world_graph_1.iplot(x='ConfirmedCases',y='Fatalities', title="World",xTitle='Confirmed Cases',yTitle='Fatalities')

world_graph_2 = df_original[['ConfirmedCases','Fatalities']]
world_graph_2.iplot(title="World")
world_graph_2 = df_original[['ConfirmedCases','Fatalities','Date']]
world_graph_2.iplot(title="World",x='Date')
world_graph_3 = df_original[['ConfirmedCases','Fatalities','Date']]
world_graph_3.iplot(title="World",x='Date',mode='markers')
world_graph_4 = df_original[['ConfirmedCases','Fatalities','Date']]
world_graph_4.iplot(kind='bar',title="World",x='Date')
df_temp_1 = df_original[['Date','ConfirmedCases','Fatalities']]
month_dict = {1:'January' , 2:'Februrary',3:'March',4:'April',5:'May',6:'June',7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'}
df_temp_1['Month'] = pd.DatetimeIndex(df_temp_1['Date']).month
df_temp_1['Month'] = df_temp_1['Month'].map(month_dict) 
df_temp_1 = df_temp_1.drop('Date',axis=1)
x = df_temp_1['ConfirmedCases'].values
y = df_temp_1['Fatalities'].values
z = (y/x)*100
z = np.round(z)
z = np.nan_to_num(z)
k = 100-z
df_temp_1['Death_Ratio'] = z
df_temp_1['Recovery_Rate'] = k
df_temp_1
df_temp_1.iplot(kind='bar',x='Month')
df_temp_1.iplot(kind='bar',x='Month',barmode='stack',bargap=0.5)
df_temp_1.iplot(kind='barh',x='Month',barmode='stack',bargap=0.7)
countries = []
Confirmed_Cases = []
Deaths = []

for x in df_original['Country_Region'].unique():
    countries.append(x)
    k_1 = df_original[df_original['Country_Region']==x]
    Confirmed_Cases.append(k_1['ConfirmedCases'].sum())
    Deaths.append(k_1['Fatalities'].sum())

nd_array_1 = np.array(Confirmed_Cases)
nd_array_2 = np.array(Deaths)

nd_array_3 = (nd_array_2/nd_array_1)*100
nd_array_3 = np.round(nd_array_3,2)
Death_Ratio = list(nd_array_3)
Recovery_Chances = 100-nd_array_3

countries_df = pd.DataFrame({'Countries':countries,'ConfirmedCases':Confirmed_Cases,'Deaths':Deaths,'Death_Ratio':Death_Ratio,'Recovery_Chances':Recovery_Chances})
countries_df
len(countries_df)
chart_1 = countries_df.iloc[0:10]
chart_2 = countries_df.iloc[10:20]
chart_3 = countries_df.iloc[-10:-1]
chart_4 = countries_df.iloc[-100:-50]
chart_1.iplot(kind='line',x='Countries')
chart_2.iplot(kind='line',x='Countries')
chart_3.iplot(kind='line',x='Countries')
chart_4.iplot(kind='line',x='Countries')
import plotly.express as px
fig = px.pie(countries_df.iloc[0:20], values='ConfirmedCases', names='Countries', title='Confirmed Cases Count')
fig.show()
fig = px.pie(countries_df.iloc[0:20], values='Deaths', names='Countries', title='Deaths Count')
fig.show()
fig = px.pie(countries_df.iloc[0:20], values='Death_Ratio', names='Countries', title='Death Ratio Count')
fig.show()
fig = px.pie(countries_df.iloc[0:20], values='Recovery_Chances', names='Countries', title='Recovery Chances Count')
fig.show()
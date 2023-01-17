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
# importing important libraries

import pandas as pd



# For Visualisation Purpose

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

import folium 

from folium import plugins



# Manipulating the default plot size

plt.rcParams['figure.figsize'] = 10, 12



# Disable warnings 

import warnings

warnings.filterwarnings('ignore')


#Reading Pakistan's COVID data

df= pd.read_excel("../input/pakistanallcitiescovid19/COVID_Pak_finaldayData.xlsx")

df_pak = df.copy()

df.head()
#Removing Nan Values

df_Pak_All = df[0:7][:] 

df_Pak_All
df_Pak_All = df_Pak_All[['Date','Region','Cumulative  Test positive','Expired','Discharged','New  (last 24 hrs)']]



df_Pak_All.head(8)  #// 8 regions
total_cases = df_Pak_All['Cumulative  Test positive'].sum()

print('Total number of confirmed COVID 2019 cases across Pakistan till date (20 Aril, 2020):', total_cases)
#Highlighten the data fram

df_Pak_All.style.background_gradient(cmap='Reds')
#Total Active  = test positive - (expired + discharged)



df_Pak_All['Total Active'] = df_Pak_All['Cumulative  Test positive'] - (df_Pak_All['Expired'] + df_Pak_All['Discharged'])

total_active = df_Pak_All['Total Active'].sum()

print('Total number of active COVID 2019 cases across Pakistan:', total_active)

Tot_Cases = df_Pak_All.groupby('Region')['Total Active'].sum().sort_values(ascending=False).to_frame()

Tot_Cases.style.background_gradient(cmap='Reds')
#Createing a zoomable map for pakistan



Pak_coord = pd.read_excel('../input/pakistanregionscordinates/Pak_coord.xlsx')



df_full = pd.merge(Pak_coord,df_Pak_All,on='Region')

map = folium.Map(location=[20, 70], zoom_start=4,tiles='Stamenterrain')



for lat, lon, value, name in zip(df_full['Latitude'], df_full['Longitude'], df_full['Total Active'], df_full['Region']):

    folium.CircleMarker([lat, lon], radius=value*0.02,color='red',fill_color='red',fill_opacity=0.3 ).add_to(map)

map
#visualization through seaborn



f, ax = plt.subplots(figsize=(12, 8))

data = df_full[['Region','Cumulative  Test positive','Discharged','Expired']]

data.sort_values('Cumulative  Test positive',ascending=False,inplace=True)

data

sns.set_color_codes("pastel")

sns.barplot(x="Cumulative  Test positive", y="Region", data=data,label="Total", color="r")



sns.set_color_codes("muted")

sns.barplot(x="Discharged", y="Region", data=data, label="Discharged", color="g")





# Add a legend and informative axis label

ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(xlim=(0, 3500), ylabel="",xlabel="Cases")

sns.despine(left=True, bottom=True)
Pak_df_dayByDay= pd.read_excel('../input/corona-virus-pakistan-dataset-2020/COVID_FINAL_DATA.xlsx')

Pak_df_dayByDay.head()
temp_total_date_Sindh = Pak_df_dayByDay[(Pak_df_dayByDay['Region']=='Sindh')].groupby(['Date']).agg({'Cumulative  Test positive':['sum']})



temp_total_date_Punjab = Pak_df_dayByDay[(Pak_df_dayByDay['Region']=='Punjab')].groupby(['Date']).agg({'Cumulative  Test positive':['sum']})



temp_total_date_Balochistan = Pak_df_dayByDay[(Pak_df_dayByDay['Region']=='Balochistan')].groupby(['Date']).agg({'Cumulative  Test positive':['sum']})



temp_total_date_ICT = Pak_df_dayByDay[(Pak_df_dayByDay['Region']=='ICT')].groupby(['Date']).agg({'Cumulative  Test positive':['sum']})



temp_total_date_KPK = Pak_df_dayByDay[(Pak_df_dayByDay['Region']=='KP')].groupby(['Date']).agg({'Cumulative  Test positive':['sum']})



temp_total_date_GB = Pak_df_dayByDay[(Pak_df_dayByDay['Region']=='GB')].groupby(['Date']).agg({'Cumulative  Test positive':['sum']})



temp_total_date_KPTD = Pak_df_dayByDay[(Pak_df_dayByDay['Region']=='KPTD')].groupby(['Date']).agg({'Cumulative  Test positive':['sum']})
Sindh = [i for i in temp_total_date_Sindh["Cumulative  Test positive"]['sum'].values]

Sindh_60 = Sindh[0:60]



Punjab = [i for i in temp_total_date_Punjab["Cumulative  Test positive"]['sum'].values]

Punjab_60 = Punjab[0:60]



Balochistan = [i for i in temp_total_date_Balochistan["Cumulative  Test positive"]['sum'].values]

Balochistan_60 = Balochistan[0:60]



ICT = [i for i in temp_total_date_ICT["Cumulative  Test positive"]['sum'].values]

ICT_60 = ICT[0:60]



KPK = [i for i in temp_total_date_KPK["Cumulative  Test positive"]['sum'].values]

KPK_60 = KPK[0:60]



GB = [i for i in temp_total_date_GB["Cumulative  Test positive"]['sum'].values]

GB_60 = GB[0:60]

# Plots

plt.figure(figsize=(12,6))



plt.plot(Sindh_60)

plt.plot(Punjab_60)

plt.plot(Balochistan_60)

plt.plot(ICT_60)

plt.plot(KPK_60)

plt.plot(GB_60)



plt.legend(["Sindh","Punjab","Balochistan","ICT","KPK","GB"], loc='upper left')

plt.title("COVID-19 infections In different regions of Pakistan", size=15)

plt.xlabel("Days", size=13)

plt.ylabel("Infected cases", size=13)

plt.ylim(0, 4000)

plt.show()
Final_Data_Pak = pd.read_excel('../input/pakistanregioninpandemic/Pakistan_Data_RegionWise.xlsx')

Final_Data_Pak.head()
Final_Data_Pak['ConfirmedCasesInPakistan'] = Final_Data_Pak['Prv_Sindh_ConfimedCases'] + Final_Data_Pak['Prv_Punjab_ConfimedCases'] + Final_Data_Pak['Prv_Balochistan_ConfimedCases'] + Final_Data_Pak['Prv_ICT_ConfimedCases']+Final_Data_Pak['Prv_GB_ConfimedCases']+Final_Data_Pak['Prv_KP_ConfimedCases']
Final_Data_Pak['NewCasesInPakistan'] = Final_Data_Pak['Prv_Sindh_NewCases'] + Final_Data_Pak['Prv_Punjab_NewCases'] + Final_Data_Pak['Prv_Balochistan_NewCases'] + Final_Data_Pak['Prv_ICT_NewCases']+Final_Data_Pak['Prv_GB_NewCases']+Final_Data_Pak['Prv_KP_NewCases']
Final_Data_Pak.head()
import plotly
Final_Data_Pak['Date'] =pd.to_datetime(Final_Data_Pak.Date)

Final_Data_Pak['Date'] = Final_Data_Pak.sort_values(by='Date')

fig = go.Figure()

fig.add_trace(go.Scatter(x=Final_Data_Pak['Date'].head(35), y = Final_Data_Pak['ConfirmedCasesInPakistan'].head(35), mode='lines+markers',name='ConfirmedCasesInPakistan'))

fig.update_layout(title_text='Trend of Coronavirus Cases in Pakistan (Cumulative cases)',plot_bgcolor='rgb(230, 230, 230)')

fig.show()



# New COVID-19 cases reported daily in Pakistan



import plotly.express as px

fig = px.bar(Final_Data_Pak, x="Date", y="NewCasesInPakistan", barmode='group', height=400)

fig.update_layout(title_text='Coronavirus Cases in Pakistan on daily basis',plot_bgcolor='rgb(230, 230, 230)')



fig.show()


dbd_Italy = pd.read_excel('../input/othercities-pandemic/per_day_cases.xlsx',parse_dates=True, sheet_name="Italy")

dbd_Korea = pd.read_excel('../input/othercities-pandemic/per_day_cases.xlsx',parse_dates=True, sheet_name="Korea")

dbd_Wuhan = pd.read_excel('../input/othercities-pandemic/per_day_cases.xlsx',parse_dates=True, sheet_name="Wuhan")
fig = px.bar(Final_Data_Pak, x="Date", y="ConfirmedCasesInPakistan", color='ConfirmedCasesInPakistan', orientation='v', height=600,

             title='Confirmed Cases in Pakistan', color_discrete_sequence = px.colors.cyclical.IceFire)



fig.update_layout(plot_bgcolor='rgb(230, 230, 230)')

fig.show()



fig = px.bar(dbd_Italy, x="Date", y="Total Cases", color='Total Cases', orientation='v', height=600,

             title='Confirmed Cases in Italy', color_discrete_sequence = px.colors.cyclical.IceFire)



fig.update_layout(plot_bgcolor='rgb(230, 230, 230)')

fig.show()



fig = px.bar(dbd_Korea, x="Date", y="Total Cases", color='Total Cases', orientation='v', height=600,

             title='Confirmed Cases in South Korea', color_discrete_sequence = px.colors.cyclical.IceFire)



fig.update_layout(plot_bgcolor='rgb(230, 230, 230)')

fig.show()

fig = px.bar(dbd_Wuhan, x="Date", y="Total Cases", color='Total Cases', orientation='v', height=600,

             title='Confirmed Cases in Wuhan', color_discrete_sequence = px.colors.cyclical.IceFire)



fig.update_layout(plot_bgcolor='rgb(230, 230, 230)')

fig.show()
from fbprophet import Prophet
confirmed_cases = Final_Data_Pak.groupby('Date').sum()['ConfirmedCasesInPakistan'].reset_index()
confirmed_cases.columns = ['ds','y']

confirmed_cases['ds'] = pd.to_datetime(confirmed_cases['ds'])
confirmed_cases.tail()
m = Prophet(interval_width=0.95)

m.fit(confirmed_cases)

future = m.make_future_dataframe(periods=7)

future.tail()
#predicting the future with date, and upper and lower limit of y value

forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
confirmed_forecast_plot = m.plot(forecast)
confirmed_forecast_plot =m.plot_components(forecast)
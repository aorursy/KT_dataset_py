!pip install fbprophet

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
pd.set_option('display.max_rows', None)
import datetime
from plotly.subplots import make_subplots

import requests
from bs4 import BeautifulSoup
from fbprophet import Prophet
import seaborn as sns
import plotly.express as px

data = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
#data3 = pd.read_csv('../kaggle/geniusdacov/osb-enftransmcovid19-1.csv')

#Percentage of NAN Values 
NAN = [(c, data[c].isna().mean()*100) for c in data]
NAN = pd.DataFrame(NAN, columns=["column_name", "percentage"])
NAN
data["Province/State"]= data["Province/State"].fillna('Unknown')

data[["Confirmed","Deaths","Recovered"]] =data[["Confirmed","Deaths","Recovered"]].astype(int)
data['Active_case'] = data['Confirmed'] - data['Deaths'] - data['Recovered']
data.head()
Data_col = data [(data['Country/Region'] == 'Colombia') ].reset_index(drop=True)

Data_col['Month'] = pd.DatetimeIndex(Data_col['ObservationDate']).month
Data_col['Day'] = pd.DatetimeIndex(Data_col['ObservationDate']).day

fig = plt.figure()
ax1 = fig.add_subplot(111)


Data_col.groupby('Month')[['Active_case']].mean().plot(ax=ax1, marker = 'o', color = 'r')
Data_col.groupby('Month')[['Deaths']].mean().plot(ax=ax1, marker = '^', color = 'b')
Data_col.groupby('Month')[['Recovered']].mean().plot(ax=ax1, marker = '', color = 'g')
Data_col.groupby('Month')[['Confirmed']].mean().plot(ax=ax1, marker = '', color = 'y')


ax1.set_title('Average comparative Per Month')

plt.figure(figsize=(12,7))




def prediction(ID, df, periods,columns):
  # Function that takes in the data frame, storeID, and number of future period forecast
  # The function then generates date/columns columns in Prophet format
  # The function then makes time series predictions

  df = df[ df['Province/State'] == ID ]
  df = df[['ObservationDate', columns]].rename(columns = {'ObservationDate': 'ds', columns:'y'})
  df = df.sort_values('ds')
  city = ID + '' + columns
  model    = Prophet()
  
  model.fit(df)
  future   = model.make_future_dataframe(periods=periods)
  forecast = model.predict(future)
  figure    = model.plot(forecast, xlabel='Fecha de observación', ylabel=city)
  str =  columns + '_' + ID + '_prophetplot.png'
  figure.savefig(str)
  figure2  = model.plot_components(forecast)
  str =  columns + '_trend_' + ID + '_prophetplot.png'
  figure2.savefig(str)
 
    
Data_col
prediction('Capital District', Data_col, 60,'Recovered')
prediction('Capital District', Data_col, 60,'Active_case')
prediction('Capital District', Data_col, 60,'Confirmed') 
prediction('Capital District', Data_col, 60,'Deaths')


Data_ven = data [(data['Country/Region'] == 'Venezuela') ].reset_index(drop=True)

Data_ven['Month'] = pd.DatetimeIndex(Data_ven['ObservationDate']).month
Data_ven['Day'] = pd.DatetimeIndex(Data_ven['ObservationDate']).day

fig = plt.figure()
ax1 = fig.add_subplot(111)


Data_ven.groupby('Month')[['Active_case']].mean().plot(ax=ax1, marker = 'o', color = 'r')
Data_ven.groupby('Month')[['Deaths']].mean().plot(ax=ax1, marker = '^', color = 'b')
Data_ven.groupby('Month')[['Recovered']].mean().plot(ax=ax1, marker = '', color = 'g')
Data_ven.groupby('Month')[['Confirmed']].mean().plot(ax=ax1, marker = '', color = 'y')

Data_ven['Province/State'] = Data_ven['Province/State'].apply(lambda x: 'Venezuela' if x == 'Unknown' else '')


ax1.set_title('Average comparative Per Month')

plt.figure(figsize=(12,7))

Data_ven
prediction('Venezuela', Data_ven, 60,'Recovered')
prediction('Venezuela', Data_ven, 60,'Active_case')
prediction('Venezuela', Data_ven, 60,'Confirmed') 
prediction('Venezuela', Data_ven, 60,'Deaths')
Data_per = data [(data['Country/Region'] == 'Peru') ].reset_index(drop=True)

Data_per['Month'] = pd.DatetimeIndex(Data_per['ObservationDate']).month
Data_per['Day'] = pd.DatetimeIndex(Data_per['ObservationDate']).day

fig = plt.figure()
ax1 = fig.add_subplot(111)


Data_per.groupby('Month')[['Active_case']].mean().plot(ax=ax1, marker = 'o', color = 'r')
Data_per.groupby('Month')[['Deaths']].mean().plot(ax=ax1, marker = '^', color = 'b')
Data_per.groupby('Month')[['Recovered']].mean().plot(ax=ax1, marker = '', color = 'g')
Data_per.groupby('Month')[['Confirmed']].mean().plot(ax=ax1, marker = '', color = 'y')


ax1.set_title('Average comparative Per Month')

plt.figure(figsize=(12,7))

prediction('Lima', Data_per, 60,'Recovered')
prediction('Lima', Data_per, 60,'Active_case')
prediction('Lima', Data_per, 60,'Confirmed') 
prediction('Lima', Data_per, 60,'Deaths')
Data_ecu = data [(data['Country/Region'] == 'Ecuador') ].reset_index(drop=True)

Data_ecu['Month'] = pd.DatetimeIndex(Data_ecu['ObservationDate']).month
Data_ecu['Day'] = pd.DatetimeIndex(Data_ecu['ObservationDate']).day

fig = plt.figure()
ax1 = fig.add_subplot(111)


Data_ecu.groupby('Month')[['Active_case']].mean().plot(ax=ax1, marker = 'o', color = 'r')
Data_ecu.groupby('Month')[['Deaths']].mean().plot(ax=ax1, marker = '^', color = 'b')
Data_ecu.groupby('Month')[['Recovered']].mean().plot(ax=ax1, marker = '', color = 'g')
Data_ecu.groupby('Month')[['Confirmed']].mean().plot(ax=ax1, marker = '', color = 'y')
Data_ecu['Province/State'] = Data_ecu['Province/State'].apply(lambda x: 'Ecuador' if x == 'Unknown' else '')


ax1.set_title('Average comparative Per Month Ecuador')

plt.figure(figsize=(12,7))
prediction('Ecuador', Data_ecu, 60,'Recovered')
prediction('Ecuador', Data_ecu, 60,'Active_case')
prediction('Ecuador', Data_ecu, 60,'Confirmed') 
prediction('Ecuador', Data_ecu, 60,'Deaths')
Data_bol = data [(data['Country/Region'] == 'Bolivia') ].reset_index(drop=True)

Data_bol['Month'] = pd.DatetimeIndex(Data_bol['ObservationDate']).month
Data_bol['Day'] = pd.DatetimeIndex(Data_bol['ObservationDate']).day

fig = plt.figure()
ax1 = fig.add_subplot(111)


Data_bol.groupby('Month')[['Active_case']].mean().plot(ax=ax1, marker = 'o', color = 'r')
Data_bol.groupby('Month')[['Deaths']].mean().plot(ax=ax1, marker = '^', color = 'b')
Data_bol.groupby('Month')[['Recovered']].mean().plot(ax=ax1, marker = '', color = 'g')
Data_bol.groupby('Month')[['Confirmed']].mean().plot(ax=ax1, marker = '', color = 'y')
Data_bol['Province/State'] = Data_bol['Province/State'].apply(lambda x: 'Bolivia' if x == 'Unknown' else '')


ax1.set_title('Average comparative Per Month B')

plt.figure(figsize=(12,7))
prediction('Bolivia', Data_bol, 60,'Recovered')
prediction('Bolivia', Data_bol, 60,'Active_case')
prediction('Bolivia', Data_bol, 60,'Confirmed') 
prediction('Bolivia', Data_bol, 60,'Deaths')
Data_bras = data [(data['Country/Region'] == 'Brazil') ].reset_index(drop=True)

Data_bras['Month'] = pd.DatetimeIndex(Data_bras['ObservationDate']).month
Data_bras['Day'] = pd.DatetimeIndex(Data_bras['ObservationDate']).day

fig = plt.figure()
ax1 = fig.add_subplot(111)


Data_bras.groupby('Month')[['Active_case']].mean().plot(ax=ax1, marker = 'o', color = 'r')
Data_bras.groupby('Month')[['Deaths']].mean().plot(ax=ax1, marker = '^', color = 'b')
Data_bras.groupby('Month')[['Recovered']].mean().plot(ax=ax1, marker = '', color = 'g')
Data_bras.groupby('Month')[['Confirmed']].mean().plot(ax=ax1, marker = '', color = 'y')


ax1.set_title('Sum comparative Per Month Brasil')

plt.figure(figsize=(12,7))


Data_bras
prediction('Sao Paulo', Data_bras, 60,'Recovered')
prediction('Sao Paulo', Data_bras, 60,'Active_case')
prediction('Sao Paulo', Data_bras, 60,'Confirmed') 
prediction('Sao Paulo', Data_bras, 60,'Deaths')

Data_urug = data [(data['Country/Region'] == 'Uruguay') ].reset_index(drop=True)

Data_urug['Month'] = pd.DatetimeIndex(Data_urug['ObservationDate']).month
Data_urug['Day'] = pd.DatetimeIndex(Data_urug['ObservationDate']).day

fig = plt.figure()
ax1 = fig.add_subplot(111)


Data_urug.groupby('Month')[['Active_case']].mean().plot(ax=ax1, marker = 'o', color = 'r')
Data_urug.groupby('Month')[['Deaths']].mean().plot(ax=ax1, marker = '^', color = 'b')
Data_urug.groupby('Month')[['Recovered']].mean().plot(ax=ax1, marker = '', color = 'g')
Data_urug.groupby('Month')[['Confirmed']].mean().plot(ax=ax1, marker = '', color = 'y')
Data_urug['Province/State'] = Data_urug['Province/State'].apply(lambda x: 'Uruguay' if x == 'Unknown' else '')

ax1.set_title('Sum comparative Per Month Uruguay')

plt.figure(figsize=(12,7))

prediction('Uruguay', Data_urug, 60,'Recovered')
prediction('Uruguay', Data_urug, 60,'Active_case')
prediction('Uruguay', Data_urug, 60,'Confirmed') 
prediction('Uruguay', Data_urug, 60,'Deaths')
Data_para = data [(data['Country/Region'] == 'Paraguay') ].reset_index(drop=True)

Data_para['Month'] = pd.DatetimeIndex(Data_para['ObservationDate']).month
Data_para['Day'] = pd.DatetimeIndex(Data_para['ObservationDate']).day

fig = plt.figure()
ax1 = fig.add_subplot(111)


Data_para.groupby('Month')[['Active_case']].mean().plot(ax=ax1, marker = 'o', color = 'r')
Data_para.groupby('Month')[['Deaths']].mean().plot(ax=ax1, marker = '^', color = 'b')
Data_para.groupby('Month')[['Recovered']].mean().plot(ax=ax1, marker = '', color = 'g')
Data_para.groupby('Month')[['Confirmed']].mean().plot(ax=ax1, marker = '', color = 'y')
Data_para['Province/State'] = Data_para['Province/State'].apply(lambda x: 'Paraguay' if x == 'Unknown' else '')

ax1.set_title('Sum comparative Per Month Paraguay')

plt.figure(figsize=(12,7))
Data_para
prediction('Paraguay', Data_para, 60,'Recovered')
prediction('Paraguay', Data_para, 60,'Active_case')
prediction('Paraguay', Data_para, 60,'Confirmed') 
prediction('Paraguay', Data_para, 60,'Deaths')
Data_arge = data [(data['Country/Region'] == 'Argentina') ].reset_index(drop=True)

Data_arge['Month'] = pd.DatetimeIndex(Data_arge['ObservationDate']).month
Data_arge['Day'] = pd.DatetimeIndex(Data_arge['ObservationDate']).day

fig = plt.figure()
ax1 = fig.add_subplot(111)


Data_arge.groupby('Month')[['Active_case']].mean().plot(ax=ax1, marker = 'o', color = 'r')
Data_arge.groupby('Month')[['Deaths']].mean().plot(ax=ax1, marker = '^', color = 'b')
Data_arge.groupby('Month')[['Recovered']].mean().plot(ax=ax1, marker = '', color = 'g')
Data_arge.groupby('Month')[['Confirmed']].mean().plot(ax=ax1, marker = '', color = 'y')
Data_arge['Province/State'] = Data_arge['Province/State'].apply(lambda x: 'Argentina' if x == 'Unknown' else '')

ax1.set_title('Sum comparative Per Month Argentina')

plt.figure(figsize=(12,7))
Data_arge
prediction('Argentina', Data_arge, 60,'Recovered')
prediction('Argentina', Data_arge, 60,'Active_case')
prediction('Argentina', Data_arge, 60,'Confirmed') 
prediction('Argentina', Data_arge, 60,'Deaths')
Data_chile = data [(data['Country/Region'] == 'Chile') ].reset_index(drop=True)

Data_chile['Month'] = pd.DatetimeIndex(Data_chile['ObservationDate']).month
Data_chile['Day'] = pd.DatetimeIndex(Data_chile['ObservationDate']).day

fig = plt.figure()
ax1 = fig.add_subplot(111)


Data_chile.groupby('Month')[['Active_case']].mean().plot(ax=ax1, marker = 'o', color = 'r')
Data_chile.groupby('Month')[['Deaths']].mean().plot(ax=ax1, marker = '^', color = 'b')
Data_chile.groupby('Month')[['Recovered']].mean().plot(ax=ax1, marker = '', color = 'g')
Data_chile.groupby('Month')[['Confirmed']].mean().plot(ax=ax1, marker = '', color = 'y')


ax1.set_title('Sum comparative Per Month Chile')

plt.figure(figsize=(12,7))



Data_chile

prediction('Antofagasta', Data_chile, 60,'Recovered')
prediction('Antofagasta', Data_chile, 60,'Active_case')
prediction('Antofagasta', Data_chile, 60,'Confirmed') 
prediction('Antofagasta', Data_chile, 60,'Deaths')
Data_col_last = Data_col[Data_col['ObservationDate'] == max(Data_col['ObservationDate'])].reset_index()
Data_col_last



left_df3        = Data_col[Data_col['Province/State'] == 'Capital District']
left_df3a        = Data_col[Data_col['Province/State'] == 'Atlantico']
left_df4        = Data_col[Data_col['Province/State'] == 'Valle del Cauca']
left_df4b        = Data_col[Data_col['Province/State'] == 'Antioquia']
left_df5        = Data_col[Data_col['Province/State'] == 'Bolivar']
left_df6        = Data_col[Data_col['Province/State'] == 'Cundinamarca']
left_df7        = Data_col[Data_col['Province/State'] == 'Narino']
left_df8        = Data_col[Data_col['Province/State'] == 'Magdalena']
left_df9        = Data_col[Data_col['Province/State'] == 'Sucre']
left_df9a        = Data_col[Data_col['Province/State'] == 'Choco']


plt.figure(figsize=(12,7))

sns.kdeplot(left_df3['Active_case'], label = 'Active cases Bogota', shade = True, color = 'black')
sns.kdeplot(left_df3a['Active_case'], label = 'Active cases Atlantico', shade = True, color = 'orange')
sns.kdeplot(left_df4['Active_case'], label = 'Active cases Valle del Cauca', shade = True, color = 'b')
sns.kdeplot(left_df4b['Active_case'], label = 'Active cases Antioquia', shade = True, color = 'g')
sns.kdeplot(left_df5['Active_case'], label = 'Active cases Bolivar', shade = True, color = 'c')
sns.kdeplot(left_df6['Active_case'], label = 'Active cases Cundinamarca', shade = True, color = 'm')
sns.kdeplot(left_df7['Active_case'], label = 'Active cases Narino', shade = True, color = 'y')
sns.kdeplot(left_df8['Active_case'], label = 'Active cases Magdalena', shade = True, color = 'silver')
sns.kdeplot(left_df9['Active_case'], label = 'Active cases Sucre', shade = True, color = 'plum')
sns.kdeplot(left_df9a['Active_case'], label = 'Active cases Choco', shade = True, color = 'cyan')






plt.xlabel('Active_case Top ten colombia Comparative by city')





plt.figure(figsize=(12,7))

sns.kdeplot(left_df3['Deaths'], label = 'Deaths cases Bogota', shade = True, color = 'black')
sns.kdeplot(left_df3a['Deaths'], label = 'Deaths Atlantico', shade = True, color = 'orange')
sns.kdeplot(left_df4['Deaths'], label = 'Deaths cases Valle del Cauca', shade = True, color = 'b')
sns.kdeplot(left_df4b['Deaths'], label = 'Deaths cases Antioquia', shade = True, color = 'g')
sns.kdeplot(left_df5['Deaths'], label = 'Deaths cases Bolivar', shade = True, color = 'c')
sns.kdeplot(left_df6['Deaths'], label = 'Deaths cases Cundinamarca', shade = True, color = 'm')
sns.kdeplot(left_df7['Deaths'], label = 'Deaths cases Narino', shade = True, color = 'y')
sns.kdeplot(left_df8['Deaths'], label = 'Deaths cases Magdalena', shade = True, color = 'silver')
sns.kdeplot(left_df9['Deaths'], label = 'Deaths cases Sucre', shade = True, color = 'plum')
sns.kdeplot(left_df9a['Deaths'], label = 'Deaths cases Choco', shade = True, color = 'cyan')




plt.xlabel('Deaths Top ten colombia Comparative by city')








plt.figure(figsize=(12,7))

sns.kdeplot(left_df3['Recovered'], label = 'Recovered cases Bogota', shade = True, color = 'black')
sns.kdeplot(left_df3['Recovered'], label = 'Recovered cases Atlantico', shade = True, color = 'orange')

sns.kdeplot(left_df4['Recovered'], label = 'Recovered cases Valle del Cauca', shade = True, color = 'b')
sns.kdeplot(left_df4b['Recovered'], label = 'Recovered cases Antioquia', shade = True, color = 'g')
sns.kdeplot(left_df5['Recovered'], label = 'Recovered cases Bolivar', shade = True, color = 'c')
sns.kdeplot(left_df6['Recovered'], label = 'Recovered cases Cundinamarca', shade = True, color = 'm')
sns.kdeplot(left_df7['Recovered'], label = 'Recovered cases Narino', shade = True, color = 'y')
sns.kdeplot(left_df8['Recovered'], label = 'Recovered cases Magdalena', shade = True, color = 'silver')
sns.kdeplot(left_df9['Recovered'], label = 'Recovered cases Sucre', shade = True, color = 'plum')
sns.kdeplot(left_df9a['Recovered'], label = 'Recovered cases Choco', shade = True, color = 'cyan')




plt.xlabel('Recovered Top ten colombia Comparative by city')
Data_col_last3 = Data_col_last.groupby(["Country/Region"])["Confirmed","Deaths","Recovered","Active_case"].sum().reset_index().reset_index(drop=True)
Data_col_last3
labels = ["Active cases","Recovered","Deaths"]
values = Data_col_last3.loc[0, ["Active_case","Recovered","Deaths"]]
df = px.data.tips()
fig = px.pie(Data_col_last3, values=values, names=labels, color_discrete_sequence=['royalblue','green','darkblue'], hole=0.5)
fig.update_layout(
    title='Total cases in Colombia : '+str(Data_col_last3["Confirmed"][0]),
)






Data_col_per_state = Data_col_last.groupby(["Province/State"])["Confirmed","Deaths","Recovered","Active_case"].sum().reset_index().sort_values("Confirmed",ascending=False).reset_index(drop=True)
Data_col_per_state
correlations = Data_col_per_state.corr()['Active_case'].sort_values()
correlations
correlations = Data_col_per_state.corr()
f, ax = plt.subplots(figsize = (20, 20))
sns.heatmap(correlations, annot = True)
Data_col_per_state.describe()
# Average confirmed 6000, deaths 03 , recovered 2817 , active case  297

Data_col_per_state.hist(bins = 30, figsize = (20,20), color = 'r')
# Let's do the same for the Day and Month
Data_col_per_state = Data_col_last.groupby(["Province/State"])["Confirmed","Deaths","Recovered","Active_case"].sum().reset_index().sort_values("Confirmed",ascending=False).reset_index(drop=True)
Data_col_per_state






px.scatter(Data_col_per_state, x='Deaths', y='Active_case', color='Province/State', size='Deaths', title='COVID19 Total Cases growth worst affected ' )

px.scatter(Data_col_per_state, x='Confirmed', y='Active_case', color='Province/State', size='Confirmed', title='COVID19 Total Cases growth worst affected ' )



px.scatter(Data_col_per_state, x='Recovered', y='Active_case', color='Province/State', size='Recovered', title='COVID19 Total Cases growth worst affected ' )


px.scatter(Data_col_per_state, x='Active_case', y='Active_case', color='Province/State', size='Active_case', title='COVID19 Total Cases growth worst affected ' )




Data_col_op= Data_col.groupby(["ObservationDate","Country/Region"])["Confirmed","Deaths","Recovered","Active_case"].sum().reset_index().reset_index(drop=True)
Data_col_op
fig = go.Figure()
fig.add_trace(go.Scatter(x=Data_col_op.index, y=Data_col_op['Confirmed'],
                    mode='lines',
                    name='Confirmed cases'))


fig.update_layout(
    title='Evolution of Confirmed cases over time in Colombia',
        template='plotly_white'

)

fig.show()

fig = go.Figure()


fig.add_trace(go.Scatter(x=Data_col_op.index, y=Data_col_op['Active_case'],
                    mode='lines',marker_color='yellow',
                    name='Active cases',line=dict( dash='dot')))

fig.update_layout(
    title='Evolution of Acitive cases over time in Colombia',
        template='plotly_dark'

)

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=Data_col_op.index, y=Data_col_op['Recovered'],
                    mode='lines',
                    name='Recovered cases',marker_color='green'))

fig.update_layout(
    title='Evolution of Recovered cases over time in Colombia',
        template='plotly_white'

)

fig.show()
!ls  ./kaggle/working/

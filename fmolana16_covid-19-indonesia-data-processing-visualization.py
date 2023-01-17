#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import plotly.graph_objects as go
#import data
data = pd.read_csv('../input/covid19-indonesia/covid_19_indonesia_time_series_all.csv')
data.head()
#investigate data information
data.info()
#change data type
data['Date'] = pd.to_datetime(data['Date'])
#investigate data statistics
data.describe()
#check null data percentage
data.isnull().mean()*100
#check missing values or NaN percentage in each column
data.isna().mean()*100
# delete column with missing (NaN) data
# 'City or Regency' has 100% null data
data = data.drop('City or Regency',axis=1)
#inspect 'Special Status' column
data['Special Status'].value_counts()
#delete 'Special Status' column
data = data.drop('Special Status', axis=1)
#inspect 'Growth Factor of New Cases' column
data['Growth Factor of New Cases'].head(10)
# replace NaN with 0
data['Growth Factor of New Cases'] = data['Growth Factor of New Cases'].fillna(float(0))
# replace NaN with 0 for "Growth Factor of New Deaths"
data['Growth Factor of New Deaths'] = data['Growth Factor of New Deaths'].fillna(float(0))
#inspect columns with same percentage
data[['Province','Island','Time Zone']].head(10)
# replace NaN value with "Unknown"
data.update(data[['Province','Island','Time Zone']].fillna('Unknown'))
#inspect 'Total Rural Villages' column
data[['Location','Province','Total Cities','Total Urban Villages','Total Rural Villages']].head(20)
# replace NaN with 0
data['Total Rural Villages'] = data['Total Rural Villages'].fillna(float(0))
# inspect 'Total Cities' column
data[['Province','Total Active Cases','Total Cities','Total Urban Villages','Total Rural Villages']].tail(10)
# replace NaN with 0
data['Total Cities'] = data['Total Cities'].fillna(float(0))
# inspect 'Total Urban Villages' column
data[['Location','Province','Total Active Cases','Total Cities','Total Urban Villages','Total Rural Villages']].tail(40)
#replace NaN with 0
data['Total Urban Villages'].fillna(float(0), inplace=True)

# check if data is clean or no missing value
data.isna().mean()*100
# inspect for duplicate value in each column
data[['Location ISO Code', 'Location', 'Province','Country','Continent','Island']].head(20)
# drop 'Location', duplicate value with 'Province'
data.drop('Location',axis=1, inplace=True)
data.head()
# extract for Indonesia Covid-19 case 
columns = ['Date', 'Location ISO Code','New Cases','New Deaths','New Recovered','New Active Cases','Total Cases','Total Deaths','Total Recovered','Total Active Cases']
data_covid = data[data['Location ISO Code']== 'IDN']
data_covid = data_covid[columns]
data_covid.head(10)
# prepare data for visualization
data_covid_total= data_covid.groupby('Date')[['Total Cases','Total Deaths','Total Recovered','Total Active Cases']].sum().reset_index().sort_values('Date', ascending=True).reset_index(drop=True)
data_covid_total.head(10)
# plot Indonesia Covid 19 cases
fig = go.Figure()
fig.add_trace(go.Scatter(x=data_covid_total['Date'],
                         y=data_covid_total['Total Cases'],
                         mode='lines',
                         name= 'Total Cases'
                        ))

fig.add_trace(go.Scatter(x=data_covid_total['Date'],
                         y=data_covid_total['Total Active Cases'],
                         mode='lines',
                         name= 'Active Cases',
                         marker_color= 'red'
                         ))

fig.add_trace(go.Scatter(x=data_covid_total['Date'],
                         y=data_covid_total['Total Deaths'],
                         mode='lines',
                         name='Death Cases',
                         marker_color='black',
                         line=dict(dash='dot')
                        ))

fig.add_trace(go.Scatter(x=data_covid_total['Date'],
                         y=data_covid_total['Total Recovered'],
                         mode='lines',
                         name='Recovered Case',
                         marker_color='green'
                        ))

fig.update_layout(title='Indonesia Covid Cases',
                 template='plotly_white',
                 xaxis_title='Date',
                 yaxis_title='Total Cases'
                 )
fig.show()
# plot Active Cases
fig = go.Figure(go.Bar(x=data_covid_total['Date'],
                       y=data_covid_total['Total Active Cases'],
                       marker_color='red'
                      ))

fig.update_layout(title='Indonesia Active Cases',
                 template='plotly_white',
                 xaxis_title='Date',
                 yaxis_title='Total Cases'
                 )

fig.show()
# plot Death Cases
fig = go.Figure(go.Bar(x=data_covid_total['Date'],
                       y=data_covid_total['Total Deaths'],
                       marker_color='black'
                      ))
fig.update_layout(title='Indonesia Death Cases',
                  template='plotly_white',
                  xaxis_title='Date',
                  yaxis_title='Total Cases'
                 )
fig.show()
# plot Recovered Cases
fig = go.Figure(go.Bar(x=data_covid_total['Date'],
                       y=data_covid_total['Total Recovered'],
                       marker_color='green'
                      ))

fig.update_layout(title='Indonesia Recovered Cases',
                  template='plotly_white',
                  xaxis_title='Date',
                  yaxis_title='Total Case'
                 )
fig.show()
# extract for DKI Jakarta Covid-19 case 
columns = ['Date', 'Location ISO Code','New Cases','New Deaths','New Recovered','New Active Cases','Total Cases','Total Deaths','Total Recovered','Total Active Cases']
JK_covid = data[data['Location ISO Code']== 'ID-JK']
JK_covid = JK_covid[columns]
JK_covid.head(10)
# prepare data for visualization
JK_covid_total= JK_covid.groupby('Date')[['Total Cases','Total Deaths','Total Recovered','Total Active Cases']].sum().reset_index().sort_values('Date', ascending=True).reset_index(drop=True)
JK_covid_total.head(10)
# plot DKI Jakarta Covid 19 cases
fig = go.Figure()
fig.add_trace(go.Scatter(x=JK_covid_total['Date'],
                         y=JK_covid_total['Total Cases'],
                         mode='lines',
                         name= 'Total Cases'
                        ))

fig.add_trace(go.Scatter(x=JK_covid_total['Date'],
                         y=JK_covid_total['Total Active Cases'],
                         mode='lines',
                         name= 'Active Cases',
                         marker_color= 'red'
                         ))

fig.add_trace(go.Scatter(x=JK_covid_total['Date'],
                         y=JK_covid_total['Total Deaths'],
                         mode='lines',
                         name='Death Cases',
                         marker_color='black',
                         line=dict(dash='dot')
                        ))

fig.add_trace(go.Scatter(x=JK_covid_total['Date'],
                         y=JK_covid_total['Total Recovered'],
                         mode='lines',
                         name='Recovered Case',
                         marker_color='green'
                        ))

fig.update_layout(title='DKI Jakarta Covid Cases',
                 template='plotly_white',
                 xaxis_title='Date',
                 yaxis_title='Total Cases')
fig.show()
# plot Active Cases
fig = go.Figure(go.Bar(x=JK_covid_total['Date'],
                       y=JK_covid_total['Total Active Cases'],
                       marker_color='red'
                      ))

fig.update_layout(title='DKI Jakarta Active Cases',
                 template='plotly_white',
                 xaxis_title='Date',
                 yaxis_title='Total Cases'
                 )

fig.show()
# plot Death Cases
fig = go.Figure(go.Bar(x=JK_covid_total['Date'],
                       y=JK_covid_total['Total Deaths'],
                       marker_color='black'
                      ))
fig.update_layout(title='DKI Jakarta Death Cases',
                  template='plotly_white',
                  xaxis_title='Date',
                  yaxis_title='Total Cases'
                 )
fig.show()
# plot Recovered
fig = go.Figure(go.Bar(x=JK_covid_total['Date'],
                       y=JK_covid_total['Total Recovered'],
                       marker_color='green'
                      ))

fig.update_layout(title='DKI Jakarta Recovered Cases',
                  template='plotly_white',
                  xaxis_title='Date',
                  yaxis_title='Total Case'
                 )
fig.show()
# extract for Jawa Barat Covid-19 case 
columns = ['Date', 'Location ISO Code','New Cases','New Deaths','New Recovered','New Active Cases','Total Cases','Total Deaths','Total Recovered','Total Active Cases']
JB_covid = data[data['Location ISO Code']== 'ID-JB']
JB_covid = JB_covid[columns]
JB_covid.head(10)
# prepare data for visualization
JB_covid_total= JB_covid.groupby('Date')[['Total Cases','Total Deaths','Total Recovered','Total Active Cases']].sum().reset_index().sort_values('Date', ascending=True).reset_index(drop=True)
JB_covid_total.head(10)
# plot Jawa Barat Covid 19 cases
fig = go.Figure()
fig.add_trace(go.Scatter(x=JB_covid_total['Date'],
                         y=JB_covid_total['Total Cases'],
                         mode='lines',
                         name= 'Total Cases'
                        ))

fig.add_trace(go.Scatter(x=JB_covid_total['Date'],
                         y=JB_covid_total['Total Active Cases'],
                         mode='lines',
                         name= 'Active Cases',
                         marker_color= 'red'
                         ))

fig.add_trace(go.Scatter(x=JB_covid_total['Date'],
                         y=JB_covid_total['Total Deaths'],
                         mode='lines',
                         name='Death Cases',
                         marker_color='black',
                         line=dict(dash='dot')
                        ))

fig.add_trace(go.Scatter(x=JB_covid_total['Date'],
                         y=JB_covid_total['Total Recovered'],
                         mode='lines',
                         name='Recovered Case',
                         marker_color='green'
                        ))

fig.update_layout(title='Jawa Barat Covid Cases',
                 template='plotly_white',
                 xaxis_title='Date',
                 yaxis_title='Total Cases'
                 )
fig.show()
# plot Active Cases
fig = go.Figure(go.Bar(x=JB_covid_total['Date'],
                       y=JB_covid_total['Total Active Cases'],
                       marker_color='red'
                      ))

fig.update_layout(title='Jawa Barat Active Cases',
                 template='plotly_white',
                 xaxis_title='Date',
                 yaxis_title='Total Cases'
                 )

fig.show()
# plot Death Cases
fig = go.Figure(go.Bar(x=JB_covid_total['Date'],
                       y=JB_covid_total['Total Deaths'],
                       marker_color='black'
                      ))
fig.update_layout(title='Jawa Barat Death Cases',
                  template='plotly_white',
                  xaxis_title='Date',
                  yaxis_title='Total Cases'
                 )
fig.show()
# plot Recovered
fig = go.Figure(go.Bar(x=JB_covid_total['Date'],
                       y=JB_covid_total['Total Recovered'],
                       marker_color='green'
                      ))

fig.update_layout(title='Jawa Barat Recovered Cases',
                  template='plotly_white',
                  xaxis_title='Date',
                  yaxis_title='Total Case'
                 )
fig.show()
# extract for Jawa Timur Covid-19 case 
columns = ['Date', 'Location ISO Code','New Cases','New Deaths','New Recovered','New Active Cases','Total Cases','Total Deaths','Total Recovered','Total Active Cases']
JI_covid = data[data['Location ISO Code']== 'ID-JI']
JI_covid = JI_covid[columns]
JI_covid.head(10)
# prepare data for visualization
JI_covid_total= JI_covid.groupby('Date')[['Total Cases','Total Deaths','Total Recovered','Total Active Cases']].sum().reset_index().sort_values('Date', ascending=True).reset_index(drop=True)
JI_covid_total.head(10)
# plot Jawa Timur Covid 19 cases
fig = go.Figure()
fig.add_trace(go.Scatter(x=JI_covid_total['Date'],
                         y=JI_covid_total['Total Cases'],
                         mode='lines',
                         name= 'Total Cases'
                        ))

fig.add_trace(go.Scatter(x=JI_covid_total['Date'],
                         y=JI_covid_total['Total Active Cases'],
                         mode='lines',
                         name= 'Active Cases',
                         marker_color= 'red'
                         ))

fig.add_trace(go.Scatter(x=JI_covid_total['Date'],
                         y=JI_covid_total['Total Deaths'],
                         mode='lines',
                         name='Death Cases',
                         marker_color='black',
                         line=dict(dash='dot')
                        ))

fig.add_trace(go.Scatter(x=JI_covid_total['Date'],
                         y=JI_covid_total['Total Recovered'],
                         mode='lines',
                         name='Recovered Case',
                         marker_color='green'
                        ))

fig.update_layout(title='Jawa Timur Covid Cases',
                 template='plotly_white',
                 xaxis_title='Date',
                 yaxis_title='Total Cases'
                 )
fig.show()
# plot Active Cases
fig = go.Figure(go.Bar(x=JI_covid_total['Date'],
                       y=JI_covid_total['Total Active Cases'],
                       marker_color='red'
                      ))

fig.update_layout(title='Jawa Timur Active Cases',
                 template='plotly_white',
                 xaxis_title='Date',
                 yaxis_title='Total Cases'
                 )

fig.show()
# plot Death Cases
fig = go.Figure(go.Bar(x=JI_covid_total['Date'],
                       y=JI_covid_total['Total Deaths'],
                       marker_color='black'
                      ))
fig.update_layout(title='Jawa Timur Death Cases',
                  template='plotly_white',
                  xaxis_title='Date',
                  yaxis_title='Total Cases'
                 )
fig.show()
# plot Recovered
fig = go.Figure(go.Bar(x=JI_covid_total['Date'],
                       y=JI_covid_total['Total Recovered'],
                       marker_color='green'
                      ))

fig.update_layout(title='Jawa Timur Recovered Cases',
                  template='plotly_white',
                  xaxis_title='Date',
                  yaxis_title='Total Case'
                 )
fig.show()
# extract for Sulawesi Selatan Covid-19 case 
columns = ['Date', 'Location ISO Code','New Cases','New Deaths','New Recovered','New Active Cases','Total Cases','Total Deaths','Total Recovered','Total Active Cases']
SN_covid = data[data['Location ISO Code']== 'ID-SN']
SN_covid = SN_covid[columns]
SN_covid.head()
# prepare data for visualization
SN_covid_total= SN_covid.groupby('Date')[['Total Cases','Total Deaths','Total Recovered','Total Active Cases']].sum().reset_index().sort_values('Date', ascending=True).reset_index(drop=True)
SN_covid_total.head()
# plot Sulawesi Selatan Covid 19 cases
fig = go.Figure()
fig.add_trace(go.Scatter(x=SN_covid_total['Date'],
                         y=SN_covid_total['Total Cases'],
                         mode='lines',
                         name= 'Total Cases'
                        ))

fig.add_trace(go.Scatter(x=SN_covid_total['Date'],
                         y=SN_covid_total['Total Active Cases'],
                         mode='lines',
                         name= 'Active Cases',
                         marker_color= 'red'
                         ))

fig.add_trace(go.Scatter(x=SN_covid_total['Date'],
                         y=SN_covid_total['Total Deaths'],
                         mode='lines',
                         name='Death Cases',
                         marker_color='black',
                         line=dict(dash='dot')
                        ))

fig.add_trace(go.Scatter(x=SN_covid_total['Date'],
                         y=SN_covid_total['Total Recovered'],
                         mode='lines',
                         name='Recovered Case',
                         marker_color='green'
                        ))

fig.update_layout(title='Sulawesi Selatan Covid Cases',
                 template='plotly_white',
                 xaxis_title='Date',
                 yaxis_title='Total Cases')
fig.show()
# plot Active Cases
fig = go.Figure(go.Bar(x=SN_covid_total['Date'],
                       y=SN_covid_total['Total Active Cases'],
                       marker_color='red'
                      ))

fig.update_layout(title='Sulawesi Selatan Active Cases',
                 template='plotly_white',
                 xaxis_title='Date',
                 yaxis_title='Total Cases')

fig.show()
# plot Death Cases
fig = go.Figure(go.Bar(x=SN_covid_total['Date'],
                       y=SN_covid_total['Total Deaths'],
                       marker_color='black'
                      ))

fig.update_layout(title='Sulawesi Selatan Death Cases',
                  template='plotly_white',
                  xaxis_title='Date',
                  yaxis_title='Total Cases'
                 )
fig.show()
# plot Recovered
fig = go.Figure(go.Bar(x=SN_covid_total['Date'],
                       y=SN_covid_total['Total Recovered'],
                       marker_color='green'
                      ))

fig.update_layout(title='Sulawesi Selatan Recovered Cases',
                  template='plotly_white',
                  xaxis_title='Date',
                  yaxis_title='Total Case'
                 )
fig.show()
# extract for Aceh Covid-19 case 
columns = ['Date', 'Location ISO Code','New Cases','New Deaths','New Recovered','New Active Cases','Total Cases','Total Deaths','Total Recovered','Total Active Cases']
AC_covid = data[data['Location ISO Code']== 'ID-AC']
AC_covid = AC_covid[columns]
# prepare data for visualization
AC_covid_total= AC_covid.groupby('Date')[['Total Cases','Total Deaths','Total Recovered','Total Active Cases']].sum().reset_index().sort_values('Date', ascending=True).reset_index(drop=True)
# plot Aceh Covid 19 cases
fig = go.Figure()
fig.add_trace(go.Scatter(x=AC_covid_total['Date'],
                         y=AC_covid_total['Total Cases'],
                         mode='lines',
                         name= 'Total Cases'
                        ))

fig.add_trace(go.Scatter(x=AC_covid_total['Date'],
                         y=AC_covid_total['Total Active Cases'],
                         mode='lines',
                         name= 'Active Cases',
                         marker_color= 'red'
                         ))

fig.add_trace(go.Scatter(x=AC_covid_total['Date'],
                         y=AC_covid_total['Total Deaths'],
                         mode='lines',
                         name='Death Cases',
                         marker_color='black',
                         line=dict(dash='dot')
                        ))

fig.add_trace(go.Scatter(x=AC_covid_total['Date'],
                         y=AC_covid_total['Total Recovered'],
                         mode='lines',
                         name='Recovered Case',
                         marker_color='green'
                        ))

fig.update_layout(title='Aceh Covid Cases',
                 template='plotly_white',
                 xaxis_title='Date',
                 yaxis_title='Total Cases')
fig.show()
# plot Active Cases
fig = go.Figure(go.Bar(x=AC_covid_total['Date'],
                       y=AC_covid_total['Total Active Cases'],
                       marker_color='red'
                      ))

fig.update_layout(title='Aceh Active Cases',
                 template='plotly_white',
                 xaxis_title='Date',
                 yaxis_title='Total Cases')

fig.show()
# plot Death Cases
fig = go.Figure(go.Bar(x=AC_covid_total['Date'],
                       y=AC_covid_total['Total Deaths'],
                       marker_color='black'
                      ))

fig.update_layout(title='Aceh Death Cases',
                  template='plotly_white',
                  xaxis_title='Date',
                  yaxis_title='Total Cases'
                 )
fig.show()
# plot Recovered
fig = go.Figure(go.Bar(x=AC_covid_total['Date'],
                       y=AC_covid_total['Total Recovered'],
                       marker_color='green'
                      ))

fig.update_layout(title='Aceh Recovered Cases',
                  template='plotly_white',
                  xaxis_title='Date',
                  yaxis_title='Total Case'
                 )
fig.show()
# extract for Bali Covid-19 case 
columns = ['Date', 'Location ISO Code','New Cases','New Deaths','New Recovered','New Active Cases','Total Cases','Total Deaths','Total Recovered','Total Active Cases']
BA_covid = data[data['Location ISO Code']== 'ID-BA']
BA_covid = BA_covid[columns]
# prepare data for visualization
BA_covid_total= BA_covid.groupby('Date')[['Total Cases','Total Deaths','Total Recovered','Total Active Cases']].sum().reset_index().sort_values('Date', ascending=True).reset_index(drop=True)
# plot Bali Covid 19 cases
fig = go.Figure()
fig.add_trace(go.Scatter(x=BA_covid_total['Date'],
                         y=BA_covid_total['Total Cases'],
                         mode='lines',
                         name= 'Total Cases'
                        ))

fig.add_trace(go.Scatter(x=BA_covid_total['Date'],
                         y=BA_covid_total['Total Active Cases'],
                         mode='lines',
                         name= 'Active Cases',
                         marker_color= 'red'
                         ))

fig.add_trace(go.Scatter(x=BA_covid_total['Date'],
                         y=BA_covid_total['Total Deaths'],
                         mode='lines',
                         name='Death Cases',
                         marker_color='black',
                         line=dict(dash='dot')
                        ))

fig.add_trace(go.Scatter(x=BA_covid_total['Date'],
                         y=BA_covid_total['Total Recovered'],
                         mode='lines',
                         name='Recovered Case',
                         marker_color='green'
                        ))

fig.update_layout(title='Bali Covid Cases',
                 template='plotly_white',
                 xaxis_title='Date',
                 yaxis_title='Total Cases')
fig.show()
# plot Active Cases
fig = go.Figure(go.Bar(x=BA_covid_total['Date'],
                       y=BA_covid_total['Total Active Cases'],
                       marker_color='red'
                      ))

fig.update_layout(title='Bali Active Cases',
                 template='plotly_white',
                 xaxis_title='Date',
                 yaxis_title='Total Cases')

fig.show()
# plot Death Cases
fig = go.Figure(go.Bar(x=BA_covid_total['Date'],
                       y=BA_covid_total['Total Deaths'],
                       marker_color='black'
                      ))

fig.update_layout(title='Bali Death Cases',
                  template='plotly_white',
                  xaxis_title='Date',
                  yaxis_title='Total Cases'
                 )
fig.show()
# plot Recovered
fig = go.Figure(go.Bar(x=BA_covid_total['Date'],
                       y=BA_covid_total['Total Recovered'],
                       marker_color='green'
                      ))

fig.update_layout(title='Bali Recovered Cases',
                  template='plotly_white',
                  xaxis_title='Date',
                  yaxis_title='Total Case'
                 )
fig.show()
# extract for Bangka Belitung Covid-19 case 
columns = ['Date', 'Location ISO Code','New Cases','New Deaths','New Recovered','New Active Cases','Total Cases','Total Deaths','Total Recovered','Total Active Cases']
BB_covid = data[data['Location ISO Code']== 'ID-BB']
BB_covid = BB_covid[columns]
# prepare data for visualization
BB_covid_total= BB_covid.groupby('Date')[['Total Cases','Total Deaths','Total Recovered','Total Active Cases']].sum().reset_index().sort_values('Date', ascending=True).reset_index(drop=True)
# plot Bangka Belitung Covid 19 cases
fig = go.Figure()
fig.add_trace(go.Scatter(x=BB_covid_total['Date'],
                         y=BB_covid_total['Total Cases'],
                         mode='lines',
                         name= 'Total Cases'
                        ))

fig.add_trace(go.Scatter(x=BB_covid_total['Date'],
                         y=BB_covid_total['Total Active Cases'],
                         mode='lines',
                         name= 'Active Cases',
                         marker_color= 'red'
                         ))

fig.add_trace(go.Scatter(x=BB_covid_total['Date'],
                         y=BB_covid_total['Total Deaths'],
                         mode='lines',
                         name='Death Cases',
                         marker_color='black',
                         line=dict(dash='dot')
                        ))

fig.add_trace(go.Scatter(x=BB_covid_total['Date'],
                         y=BB_covid_total['Total Recovered'],
                         mode='lines',
                         name='Recovered Case',
                         marker_color='green'
                        ))

fig.update_layout(title='Bangka Belitung Covid Cases',
                 template='plotly_white',
                 xaxis_title='Date',
                 yaxis_title='Total Cases')
fig.show()
# plot Active Cases
fig = go.Figure(go.Bar(x=BB_covid_total['Date'],
                       y=BB_covid_total['Total Active Cases'],
                       marker_color='red'
                      ))

fig.update_layout(title='Bangka Belitung Active Cases',
                 template='plotly_white',
                 xaxis_title='Date',
                 yaxis_title='Total Cases')

fig.show()
# plot Death Cases
fig = go.Figure(go.Bar(x=BB_covid_total['Date'],
                       y=BB_covid_total['Total Deaths'],
                       marker_color='black'
                      ))

fig.update_layout(title='Bangka Belitung Death Cases',
                  template='plotly_white',
                  xaxis_title='Date',
                  yaxis_title='Total Cases'
                 )
fig.show()
# plot Recovered
fig = go.Figure(go.Bar(x=BB_covid_total['Date'],
                       y=BB_covid_total['Total Recovered'],
                       marker_color='green'
                      ))

fig.update_layout(title='Bangka Belitung Recovered Cases',
                  template='plotly_white',
                  xaxis_title='Date',
                  yaxis_title='Total Case'
                 )
fig.show()
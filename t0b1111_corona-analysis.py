# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# !pip install plotly==3.10.0



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime as dt

import warnings

import time

# import plotly.plotly as py

import plotly.graph_objs as go

import plotly.express as px



warnings.filterwarnings("ignore")





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_cov = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

df_cnf = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

df_rec = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

df_death = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

df_cov.drop(columns=['SNo'],inplace=True)



# print(df_cov.columns)

df_cov['ObservationDate'] = pd.to_datetime(df_cov['ObservationDate'] )

df_cov = df_cov.set_index('ObservationDate')
df_wrld = df_cov.loc[:,['Confirmed','Deaths','Recovered']]

df_wrld = df_wrld.groupby(['ObservationDate']).sum()

print(df_wrld.tail())

# df_wrld.head()

cnf_data = go.Scatter(x=df_wrld.index,hovertext='Confirm',

                         y=df_wrld.Confirmed)

dea_data = go.Scatter(x=df_wrld.index,hovertext='Death',

                         y=df_wrld.Deaths,

                     )

rec_data = go.Scatter(x=df_wrld.index,hovertext='Recovered'

                         ,y=df_wrld.Recovered,

                     )



layout = go.Layout(title='COVID-19 progression', xaxis=dict(title='Date'),

                   yaxis=dict(title='Confirem',color='blue'),

                  yaxis2=dict(title='Death', color='red',

                               overlaying='y', side='right'),

                  yaxis3=dict(title='Recovered', color='green',

                               overlaying='y', side='left'),

                  template="plotly_dark")



fig = go.Figure(data=[cnf_data,dea_data,rec_data], layout=layout)

fig.show()
cnf_period = df_cnf.drop(columns=['Province/State','Country/Region','Lat','Long']).columns

death_period = df_death.drop(columns=['Province/State','Country/Region','Lat','Long']).columns

rec_period = df_rec.drop(columns=['Province/State','Country/Region','Lat','Long']).columns



df_cnf1 = df_cnf.melt(id_vars=['Province/State','Country/Region','Lat','Long'],value_vars=cnf_period,var_name='Date',value_name='count')

df_death1 = df_death.melt(id_vars=['Province/State','Country/Region','Lat','Long'],value_vars=death_period,var_name='Date',value_name='count')

df_rec1 = df_rec.melt(id_vars=['Province/State','Country/Region','Lat','Long'],value_vars=rec_period,var_name='Date',value_name='count')





df_cnf1.dropna(subset=['count', 'Country/Region'],inplace=True) 

df_death1.dropna(subset=['count', 'Country/Region'],inplace=True) 

df_rec1.dropna(subset=['count', 'Country/Region'],inplace=True) 



fig = px.scatter_geo(df_cnf1, lat='Lat',lon='Long',color='Country/Region',

                     hover_name="Country/Region", size='count',

                     animation_frame="Date",

                     projection="natural earth",

                    title='Patient Confirm Progression ',template="plotly_dark")

# fig['data'][0].update(mode='markers+text', textposition='bottom center',

#                       text=df_cnf['Country/Region'].map('{}'.format).astype(str)+' '+\

#                       str(df_cnf['3/20/20']))





#     time.sleep(1)

    

fig.show()

fig = px.scatter_geo(df_death1, lat='Lat',lon='Long',color='Country/Region',

                     hover_name="Country/Region", size='count',

                     animation_frame="Date",

                     projection="natural earth",

                    title='Patient Death Progression ',template="plotly_dark")

# fig['data'][0].update(mode='markers+text', textposition='bottom center',

#                       text=df_cnf['Country/Region'].map('{}'.format).astype(str)+' '+\

#                       str(df_cnf['3/20/20']))





#     time.sleep(1)

    

fig.show()



fig = px.scatter_geo(df_rec1, lat='Lat',lon='Long',color='Country/Region',

                     hover_name="Country/Region", size='count',

                     animation_frame="Date",

                     projection="natural earth",

                    title='Patient Recovered progression ',template="plotly_dark")

# fig['data'][0].update(mode='markers+text', textposition='bottom center',

#                       text=df_cnf['Country/Region'].map('{}'.format).astype(str)+' '+\

#                       str(df_cnf['3/20/20']))





#     time.sleep(1)

    

fig.show()









df_cntry = df_cnf.loc[:,["Country/Region",df_cnf.columns[-1]]]

df_cntry = df_cntry.rename(columns={df_cnf.columns[-1]:'Confirm'})

df_cntry.fillna(0)

df_cntry = df_cntry.join(df_death.loc[:,[df_death.columns[-1]]]).rename(columns={df_death.columns[-1]:'Death'})

df_cntry.fillna(0)

df_cntry = df_cntry.join(df_rec.loc[:,[df_rec.columns[-1]]]).rename(columns={df_rec.columns[-1]:'Recovered'})

df_cntry.fillna(0)

df_cntry = df_cntry.groupby('Country/Region').sum()



df_cntry.sort_values(by=['Confirm'],inplace=True,ascending=False)
print(df_cntry)




fig = px.bar(df_cntry.iloc[:30],x=df_cntry.index[:30],y='Confirm',color='Death',template="plotly_dark")

fig.show()
fig = px.bar(df_cntry.iloc[:30],x=df_cntry.index[:30],y='Confirm',color='Recovered',template="plotly_dark")



fig.show()
df_contr_cnf = df_cov.groupby(['Country/Region']).max()

top_country = list(df_contr_cnf['Confirmed'].nlargest(10).index)



df_top = df_contr_cnf.loc[top_country]

df_top.reset_index(inplace = True)

df_top1 = df_top.melt(id_vars=['Country/Region','Confirmed'],value_vars=['Deaths','Recovered'],var_name='Result',value_name='count')

df_top1 = df_top1[::-1]

df_top.head(20)
fig = px.bar(df_top1,x='count',y='Country/Region',color='Result',orientation = "h",template="plotly_dark")

fig.update_traces( marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6,)

fig.show()

df_top = df_top[::-1]

fig = px.bar(df_top,x='Confirmed',y='Country/Region',orientation = "h",template="plotly_dark")

fig.update_traces( marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6,)

fig.show()
df_china_rec = df_rec[df_rec['Country/Region']=='China'].drop(columns=['Province/State','Lat',	'Long'])

df_italy_rec = df_rec[df_rec['Country/Region']=='Italy'].drop(columns=['Province/State','Lat',	'Long']).transpose()

df_iran_rec = df_rec[df_rec['Country/Region']=='Iran'].drop(columns=['Province/State','Lat',	'Long']).transpose()



df_italy_rec = df_italy_rec.rename(columns={df_italy_rec.columns[0]:'Italy'}) 

df_iran_rec = df_iran_rec.rename(columns={df_iran_rec.columns[0]:'Iran'})





df_china_rec = df_china_rec.groupby('Country/Region').sum().transpose()



df_spain_rec = df_rec[df_rec['Country/Region']=='Spain'].drop(columns=['Country/Region','Province/State','Lat',	'Long']).transpose()



df_spain_rec = df_spain_rec.rename(columns={df_spain_rec.columns[0]:'Spain'})



df_US_rec = df_rec[df_rec['Country/Region']=='US'].drop(columns=['Country/Region','Province/State','Lat',	'Long']).transpose()



df_US_rec = df_US_rec.rename(columns={df_US_rec.columns[0]:'US'})





df_iran_rec.drop(['Country/Region'],inplace=True)

df_italy_rec.drop(['Country/Region'],inplace=True)



df_top_rec = pd.concat([df_china_rec,df_italy_rec,df_iran_rec,df_spain_rec,df_US_rec], axis=1, sort=False)




df_china_cnf = df_cnf[df_cnf['Country/Region']=='China'].drop(columns=['Province/State','Lat',	'Long'])

df_italy_cnf = df_cnf[df_cnf['Country/Region']=='Italy'].drop(columns=['Country/Region','Province/State','Lat',	'Long']).transpose()

df_iran_cnf = df_cnf[df_cnf['Country/Region']=='Iran'].drop(columns=['Country/Region','Province/State','Lat',	'Long']).transpose()





df_china_cnf = df_china_cnf.groupby('Country/Region').sum().transpose()



df_italy_cnf = df_italy_cnf.rename(columns={df_italy_cnf.columns[0]:'Italy'}) 

df_iran_cnf = df_iran_cnf.rename(columns={df_iran_cnf.columns[0]:'Iran'})



df_spain_cnf = df_cnf[df_cnf['Country/Region']=='Spain'].drop(columns=['Country/Region','Province/State','Lat',	'Long']).transpose()



df_spain_cnf = df_spain_cnf.rename(columns={df_spain_cnf.columns[0]:'Spain'})



df_US_cnf = df_cnf[df_cnf['Country/Region']=='US'].drop(columns=['Country/Region','Province/State','Lat',	'Long']).transpose()



df_US_cnf = df_US_cnf.rename(columns={df_US_cnf.columns[0]:'US'})





df_top_cnf = pd.concat([df_iran_cnf,df_italy_cnf,df_china_cnf,df_spain_cnf,df_US_cnf], axis=1, sort=False)



df_china_death = df_death[df_death['Country/Region']=='China'].drop(columns=['Province/State','Lat',	'Long'])

df_italy_death= df_death[df_death['Country/Region']=='Italy'].drop(columns=['Country/Region','Province/State','Lat',	'Long']).transpose()

df_iran_death = df_death[df_death['Country/Region']=='Iran'].drop(columns=['Country/Region','Province/State','Lat',	'Long']).transpose()

df_spain_death = df_death[df_death['Country/Region']=='Spain'].drop(columns=['Country/Region','Province/State','Lat',	'Long']).transpose()

df_US_death = df_death[df_death['Country/Region']=='US'].drop(columns=['Country/Region','Province/State','Lat',	'Long']).transpose()



df_china_death = df_china_death.groupby('Country/Region').sum().transpose()



df_italy_death = df_italy_death.rename(columns={df_italy_death.columns[0]:'Italy'}) 

df_iran_death = df_iran_death.rename(columns={df_iran_death.columns[0]:'Iran'})

df_spain_death = df_spain_death.rename(columns={df_spain_death.columns[0]:'Spain'})

df_US_death = df_US_death.rename(columns={df_US_death.columns[0]:'US'})



df_top_death = pd.concat([df_iran_death,df_italy_death,df_china_death,df_spain_death,df_US_death ], axis=1, sort=False)
# cnf_data = go.Scatter(x=df_top_cnf.index,

#                          y=df_top_cnf.China,hovertext='China')

# dea_data = go.Scatter(x=df_top_cnf.index,

#                          y=df_top_cnf.Italy,hovertext='Italy'

#                      yaxis='y2')

# rec_data = go.Scatter(x=df_top_cnf.index,

#                          y=df_top_cnf.Iran,hovertext='Iran'

#                      yaxis='y3')



# layout = go.Layout(title='COVID-19 Confirm Cases progression in top three countries', xaxis=dict(title='Date'),

#                    yaxis=dict(title='Iran',color='green'),

#                   yaxis2=dict(title='Italy', color='red',

#                                overlaying='y', side='right'),

#                   yaxis3=dict(title='China', color='blue',

#                                overlaying='y', side='left'),

#                   template="plotly_dark")



# fig = go.Figure(data=[cnf_data,dea_data,rec_data], layout=layout)

# fig.show()
df_top_cnf
cnf_data = go.Scatter(x=df_top_rec.index,

                         y=df_top_rec.China,hovertext='China')

dea_data = go.Scatter(x=df_top_rec.index,

                         y=df_top_rec.Italy,hovertext='Italy',

                     yaxis='y')

rec_data = go.Scatter(x=df_top_rec.index,

                         y=df_top_rec.Iran,hovertext='Iran',

                     yaxis='y')



sp_data = go.Scatter(x=df_top_rec.index,

                         y=df_top_rec.Spain,hovertext='Spain',

                     yaxis='y')

US_data = go.Scatter(x=df_top_rec.index,

                         y=df_top_rec.US,hovertext='US',

                     yaxis='y')



layout = go.Layout(title='COVID-19 Recovered Cases progression in top countries', xaxis=dict(title='Date'),

                     yaxis=dict(title='Count',color='White'),

               

                  template="plotly_dark")



fig = go.Figure(data=[cnf_data,dea_data,rec_data,sp_data,US_data], layout=layout)

fig.show()
cnf_data = go.Scatter(x=df_top_cnf.index,

                         y=df_top_cnf.China,hovertext='China')

dea_data = go.Scatter(x=df_top_cnf.index,

                         y=df_top_cnf.Italy,hovertext='Italy',

                     yaxis='y')

rec_data = go.Scatter(x=df_top_cnf.index,

                         y=df_top_cnf.Iran,hovertext='Iran',

                     yaxis='y')

sp_data = go.Scatter(x=df_top_cnf.index,

                         y=df_top_cnf.Spain,hovertext='Spain',

                     yaxis='y')

US_data = go.Scatter(x=df_top_cnf.index,

                         y=df_top_cnf.US,hovertext='US',

                     yaxis='y')



layout = go.Layout(title='COVID-19 Confirm Cases progression in top three countries', xaxis=dict(title='Date'),

                   yaxis=dict(title='Count',color='White'),

                

                  template="plotly_dark")



fig = go.Figure(data=[cnf_data,dea_data,rec_data,sp_data,US_data], layout=layout)

fig.show()
cnf_data = go.Scatter(x=df_top_death.index,

                         y=df_top_death.China,hovertext='China')

dea_data = go.Scatter(x=df_top_death.index,

                         y=df_top_death.Italy,hovertext='Italy',

                     yaxis='y')

rec_data = go.Scatter(x=df_top_death.index,

                         y=df_top_death.Iran,hovertext='Iran',

                     yaxis='y')

sp_data = go.Scatter(x=df_top_death.index,

                         y=df_top_death.Spain,hovertext='Spain',

                     yaxis='y')

US_data = go.Scatter(x=df_top_death.index,

                         y=df_top_death.US,hovertext='US',

                     yaxis='y')



layout = go.Layout(title='COVID-19 Death progression in top three countries', xaxis=dict(title='Date'),

                   yaxis=dict(title='Count',color='White '),

                 

                  template="plotly_dark")



fig = go.Figure(data=[cnf_data,dea_data,rec_data,sp_data,US_data], layout=layout)

fig.show()
import seaborn as sns

import geopandas as gpd

import matplotlib.pyplot as plt

from plotly.subplots import make_subplots



sns.set_style('dark')



map_df = gpd.read_file('/kaggle/input/indian-states-shp-file/Indian_States.shp')

map_df.loc[0,['st_nm']] = 'Andaman and Nicobar Islands'

map_df.head()



df_india = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')

df_ind_bed =  pd.read_csv('/kaggle/input/covid19-in-india/HospitalBedsIndia.csv')

df_ind_ICMR =  pd.read_csv('/kaggle/input/covid19-in-india/ICMRTestingDetails.csv')

df_ind_indiv =  pd.read_csv('/kaggle/input/covid19-in-india/IndividualDetails.csv')

df_ind_census =  pd.read_csv('/kaggle/input/covid19-in-india/population_india_census2011.csv')





df_india = df_india.fillna(0)

df_india = df_india.replace(regex={r'-|--': 0})



convert_dict = {'ConfirmedIndianNational':int,	'ConfirmedForeignNational':int}

df_india = df_india.astype(convert_dict)



df_india['Confirmed'] = df_india['ConfirmedIndianNational']+ df_india['ConfirmedForeignNational']
df_forMap = df_india.drop(columns=['Date','Sno']).groupby('State/UnionTerritory').max()



 
df_forMap.isnull().sum()
df_forMap
merged = map_df.set_index('st_nm').join(df_forMap)

merged.dtypes
fig, ax = plt.subplots(5, figsize=(9, 45))





topic = ['Confirmed','ConfirmedIndianNational','ConfirmedForeignNational','Cured','Deaths']

cmaps = ['Oranges','Blues', 'Purples', 'Greens', 'Reds']

for i,l in enumerate(topic):

    ax[i].axis('off')

    ax[i].set_title('{} Cases of COVID 19 in India'.format(l), fontdict={'fontsize': '20', 'fontweight' : '5'})

    



    merged.plot(column=l, cmap=cmaps[i], linewidth=0.8, ax=ax[i], edgecolor='0.75', legend=True)
df_datechart = df_india.drop(columns=['State/UnionTerritory','Sno'])

df_datechart['Date'] = pd.to_datetime(df_datechart['Date'],format='%d/%m/%y')



df_datechart = df_datechart.groupby('Date').sum()





cnf_data = go.Bar(x=df_datechart.index,

                         y=df_datechart.Confirmed,hovertext='Confirmed')

dea_data = go.Bar(x=df_datechart.index,

                         y=df_datechart.Deaths,hovertext='Deaths',

                     yaxis='y2')

rec_data = go.Bar(x=df_datechart.index,

                         y=df_datechart.Cured,hovertext='Cured',

                     yaxis='y2')



layout = go.Layout(title='COVID-19 progression in India', xaxis=dict(title='Date'),

                   yaxis=dict(title='Confirem',color='blue'),

                  yaxis2=dict(title='Scale of Recovery and death', color='white',

                               overlaying='y', side='right'),

                  yaxis3=dict(title='Recovered', color='white',

                               overlaying='y', side='top'),

                  template="plotly_dark")



fig = go.Figure(data=[cnf_data,dea_data,rec_data], layout=layout)

fig.update_traces(marker_line_width=1.5, opacity=0.7)

fig.show()
df_datechart
states = df_india['State/UnionTerritory'].unique()





# fig = make_subplots(rows=6, cols=5)

# p=0

# for i in range(1:7):

#     for j in range(1:6):

#         cnf_data = go.Bar(x=df_forMap.iloc[p],

#                          y=df_datechart.iloc[p].Confirmed)

#         dea_data = go.Bar(x=df_forMap.iloc[p],

#                                  y=df_datechart.iloc[p].Deaths,

#                              yaxis='y2')

#         rec_data = go.Bar(x=df_forMap.iloc[p],

#                                  y=df_datechart.iloc[p].Cured,

#                              yaxis='y2')



#         layout = go.Layout(title=states, xaxis=dict(title='Date'),

#                            yaxis=dict(title='Confirem',color='blue'),

#                           yaxis2=dict(title='Death', color='red',

#                                        overlaying='y', side='right'),

#                           yaxis3=dict(title='Recovered', color='white',

#                                        overlaying='y', side='top'),

#                           template="plotly_dark")

#         fig.add_trace(go.Figure(data=[cnf_data,dea_data,rec_data], layout=layout)),

#           row=i, col=j)







cnf_data = go.Bar(y=df_forMap.index,

                         x=df_forMap.Confirmed,orientation='h',hovertext='Confirmed')

dea_data = go.Bar(y=df_forMap.index,

                         x=df_forMap.Deaths,

                     yaxis='y',orientation='h',hovertext='Death')

rec_data = go.Bar(y=df_forMap.index,

                         x=df_forMap.Cured,

                     yaxis='y',orientation='h',hovertext='Cured')



layout = go.Layout(title='COVID-19 data for each State', xaxis=dict(title='Count'),

                   yaxis=dict(title='States',color='White'),

                  

                  template="plotly_dark")



fig = go.Figure(data=[cnf_data,dea_data,rec_data], layout=layout)

fig.show()
df_forMap.Confirmed.sum()
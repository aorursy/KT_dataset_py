import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import folium

import seaborn as sns
df = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')

df.head()
df = df.dropna()
df = df.replace({'Telangana':'Telengana'})
df['Date'] = pd.to_datetime(df.Date,dayfirst=True).dt.strftime('%Y-%m-%d')

df['Time'] = pd.to_datetime(df['Time']).dt.strftime('%H:%M:%S')

df.tail()
df.info()
df = df.loc[df['State/UnionTerritory'] != 'Unassigned']

df = df.loc[df['State/UnionTerritory'] != 'Cases being reassigned to states']
df_train = df[['ConfirmedIndianNational','ConfirmedForeignNational','Cured','Deaths','Confirmed']]

sns.pairplot(df_train)
df_u = df['State/UnionTerritory'].unique()

df3 = pd.DataFrame()

for s in (df_u):

                l = df.loc[df['State/UnionTerritory'] == s]

                l1= l['Sno'].idxmax(axis=1)

                l2 = l.loc[l.index == l1]

                df3 = df3.append([l2])

df3.head()                                            

                                           

                                                                
df3 = df3.reset_index()

df3 = df3.drop(columns=['index','Sno'])

df3 = df3[['State/UnionTerritory','Confirmed','Deaths','Cured']]

df3 = df3.loc[df3['State/UnionTerritory'] != 'Unassigned']

df3 = df3.loc[df3['State/UnionTerritory'] != 'Cases being reassigned to states']

df3.head()
df3['Total Cases'] = (df3['Confirmed'] + df3['Deaths'] + df3['Cured']).astype(int)

df3.head()

cols = df3[['State/UnionTerritory','Total Cases']]

states = np.asarray(df3['State/UnionTerritory'])

plt.figure(figsize=(15,10))

p = sns.barplot(x=df3['State/UnionTerritory'],y=df3['Total Cases'])

p.set_xticklabels(labels=states,rotation=90)

plt.title('Total Cases in India');
import matplotlib.ticker as ticker

fig =plt.figure(figsize=(20,20));

for i,j in zip(df_u,range(1,len(df_u))):

                                                                        g = df.loc[df['State/UnionTerritory']==i]

                                                                        x = g['Date']

                                                                        y = g['Confirmed']

                                                                        ax = plt.subplot(9,4,j)

                                                                        ax.plot(x,y)

                                                                        plt.xticks(rotation=90)

                                                                        plt.title(i)

                                                                        fig.tight_layout(pad=3.0)

                                                                        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

                                                                        
df_testing = pd.read_csv('/kaggle/input/covid19-in-india/StatewiseTestingDetails.csv')

df_testing = df_testing.fillna(0)

df_testing['Date'] = pd.to_datetime(df_testing['Date']).dt.strftime('%Y-%m-%d')

df_testing.head()
fig =plt.figure(figsize=(20,20))



for i,j in zip(df_u,range(1,len(df_u))):        

                                                                        g = df_testing.loc[df_testing['State']==i]

                                                                        x = g['Date']

                                                                        y = g['TotalSamples']

                                                                        ax = plt.subplot(9,4,j)

                                                                        ax.plot(x,y,color='green')

                                                                        plt.xticks(rotation=90)

                                                                        plt.title(i)

                                                                        fig.tight_layout(pad=3.0)

                                                                        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

                                                                        

                                                                        
fig =plt.figure(figsize=(20,20))



for i,j in zip(df_u,range(1,len(df_u))):        

                                                                        g = df_testing.loc[df_testing['State']==i]

                                                                        x = g['Date']

                                                                        y = g['Positive']

                                                                        ax = plt.subplot(9,4,j)

                                                                        ax.plot(x,y,color='red')

                                                                        plt.xticks(rotation=90)

                                                                        plt.title(i)

                                                                        fig.tight_layout(pad=3.0)

                                                                        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

                                                                        
df4 = pd.DataFrame(columns=[])

for i in df_u:

            state = df.loc[df['State/UnionTerritory'] == i]

            state1 = df_testing.loc[df_testing['State'] == i]

            

            for j in state['Date']:

                                        t = state1.loc[state1['Date'] == j]

                                        t1 = state.loc[state['Date'] == j]

                                        df4 = df4.append(t1.merge(t,how='outer',on=['Date']))



                                       

                             
df4 = df4.drop(columns=['Time','ConfirmedIndianNational','ConfirmedForeignNational','State'],axis=1)

df4 = df4.fillna(0)

df4 = df4.reset_index()

df4 = df4.drop(columns = ['index','Sno'],axis=1)

df4
coff = df4.corr()

coff[['Confirmed']]
from sklearn.utils import shuffle

df5 = shuffle(df4)

df5 = df5.reset_index()

df5= df5.drop(columns=['index'],axis=1)

df5 = df5.sort_values(by='Date')

df5
import ipywidgets as widgets

from ipywidgets import interact, interact_manual

@interact

def forecast_model(State = df_u,days = 5):

                                df6 = df5.loc[df5['State/UnionTerritory'] == State]

                                df6 = df6[['Date','Confirmed','TotalSamples','Positive']]

                                df6.index = df6['Date']

                                df6 = df6.drop(columns=['Date'])

                                df6_upd = df6.loc[df6.index != df6.index.max()]

                                

                        # Fit the exisiting data trends to the forecast model

                                from statsmodels.tsa.vector_ar.vecm import coint_johansen

                                jtest = coint_johansen(df6_upd,1,1);

                                from statsmodels.tsa.vector_ar.var_model import VAR

                                m = VAR(df6_upd);

                                model = m.fit();

                            #    ax = model.plot();

                             #   ax.tight_layout(pad=3.0)

                       # predict the model         

                                valid_pred = model.forecast(model.y,steps=days);

                             #   ax = model.plot_forecast(days);

                            #    ax.tight_layout(pad=3.0)

                                df7 = pd.DataFrame(valid_pred.round(0),columns=[['Confirmed','TotalSamples','Positive']])

                                df7.index = pd.date_range(df6.index.max(),periods=days);

                                

                       #plot predictions         

                                plt.figure(figsize=(10,5))



                                plt.subplot(1,2,1)

                                plt.plot(df7[['Confirmed']],color='red')

                                plt.xticks(df7.index,rotation=90)

                                plt.tight_layout(pad=7.0)

                                plt.title('Confirmed cases')

                                plt.subplot(1,2,2)

                                plt.plot(df7[['TotalSamples']])

                                plt.xticks(df7.index,rotation=90)

                                plt.title('Total Samples to be tested')

                                return df7

                                
df8 = pd.DataFrame(columns=[['State','Confirmed','TotalSamples','Positive']])

df10 = pd.DataFrame()

df_f = pd.DataFrame()

for i in df_u:

                                df9 = df5.loc[df5['State/UnionTerritory'] == i]

                                df9 = df9[['Date','Confirmed','TotalSamples','Positive']]

                                df9.index = df9['Date']

                                df9 = df9.drop(columns=['Date'])

                                df9_upd = df9.loc[df9.index != df9.index.max()]

                                

                        # Fit the exisiting data trends to the forecast model

                                if i != 'Daman & Diu':

                                                                                

                                                                                    from statsmodels.tsa.vector_ar.var_model import VAR

                                                                                    m = VAR(df9_upd) 

                                                                                    model = m.fit()

                                                                               # predict the model         

                                                                                    valid_pred = model.forecast(model.y,steps=10);

                                                                                    df8 = pd.DataFrame(valid_pred.round(0))

                                                                                    df8.index = pd.date_range(df9.index.max(),periods=10);

                                                                                    for j in range(len(df8)):

                                                                                                                df10 = df10.append([i])

                                                                                    df10.index = pd.date_range(df9.index.max(),periods=10);

                                                                                    df_f = df_f.append(df10.merge(df8,how='outer',on=df10.index))

                                                                                    df10 = pd.DataFrame()

                                                        

                                
df_forecast = df_f

df_forecast.columns=[['Date','State','Confirmed','TotalSamples','Positive']]

df_forecast = df_forecast.reset_index()

df_forecast = df_forecast.drop(columns=['index'],axis=1);

df_forecast
df_forecast.to_csv('/kaggle/working/forecast.csv')

out = pd.read_csv('/kaggle/working/forecast.csv')

out.head()
fig =plt.figure(figsize=(20,20))



for i,j in zip(df_u,range(1,len(df_u))):        

                                                                        g = out.loc[out['State']==i]

                                                                        x = g['Date']

                                                                        y = g['Confirmed']

                                                                        ax = plt.subplot(9,4,j)

                                                                        ax.plot(x,y,color='red')

                                                                        plt.xticks(rotation=90)

                                                                        plt.title(i)

                                                                        fig.tight_layout(pad=3.0)

                                                                        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

                                                                        
df2 = pd.read_csv('/kaggle/input/states/states.csv')

df2.head()
import folium

ind_map = folium.Map(location=(20,78),zoom_state=2)

@interact

def map_view(Date=out['Date']):

                    for i in df2['State']:

                                            d = out.loc[out['State'] == i]

                                            d1 = df2.loc[df2['State'] == i]

                                            d2 = d.loc[d['Date'] == Date]



                                            label = folium.Popup('Date:'+ str(Date) + ',' + str(d2[['Confirmed']]) + ','+ str(d2[['Positive']]), parse_html=True)

                                         

                                            folium.Marker(

                                                    [d1['Latitude'],d1['Longitude']],

                                                    radius=10,

                                                    popup=label,

                                                    fill=True,

                                                    fill_opacity=0.7).add_to(ind_map)

                    return ind_map





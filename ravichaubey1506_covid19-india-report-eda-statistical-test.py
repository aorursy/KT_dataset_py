import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import plotly.graph_objects as go
cf.go_offline()
covid19Personal = pd.read_csv('../input/covid19-india/covidIndiaSimple.csv')
covid19Personal.drop('Unnamed: 0',axis = 1,inplace =True)
covid19Personal.drop([32,33,34],inplace=True)
covid19Personal['Total Confirmed cases']=covid19Personal['Total Confirmed cases'].astype('int')
covid19Personal['Cured']=covid19Personal['Cured'].astype('int')
covid19Personal['Death']=covid19Personal['Death'].astype('int')
covid19Personal.head()
covid_df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',index_col=1,parse_dates=True)
covid_df.drop(['Last Update','SNo'],axis = 1, inplace = True)
covid_df.head()
covid_india = covid_df[covid_df['Country/Region']=='India']
covid_india.index.freq = 'D'
covid_india.head()
covid_indStates = pd.read_csv('../input/covid19-in-india/covid_19_india.csv',index_col=1,parse_dates=True)
covid_indStates.drop(['Sno','Time'],axis = 1, inplace = True)
covid_indStates.head()
fig = go.Figure()
fig.add_trace(go.Scatter(x=covid_india.index, y=covid_india['Confirmed'], name='Confirmed',
                         line=dict(color='blue', width=4)))
fig.add_trace(go.Scatter(x=covid_india.index, y=covid_india['Recovered'], name='Recovered',
                         line=dict(color='green', width=4)))
fig.add_trace(go.Scatter(x=covid_india.index, y=covid_india['Deaths'], name='Deaths',
                         line=dict(color='firebrick', width=4)))

fig.update_layout(
    title='Corona Virus Trend in India',
     yaxis=dict(
        title='Number of Cases Per Day')
    )

fig.show()
covid_india[['Confirmed']].iplot(kind='bar',color=['blue'],title = 'Confirmed Cases in India',
                                 yTitle = 'Cases Per Day',xTitle='Day')
covid_india[['Recovered']].iplot(kind='bar',color=['green'],title = 'Recovered Cases in India',
                                 yTitle = 'Recovery Per Day',xTitle='Day')
covid_india[['Deaths']].iplot(kind='bar',color=['red'],title = 'Death in India',
                              yTitle = 'Death Count Per Day',xTitle='Day')
states = covid_indStates['State/UnionTerritory'].unique()
fig1 = go.Figure()
for i in states:
    fig1.add_trace(go.Scatter(x=covid_india.index,
                          y=covid_indStates[covid_indStates['State/UnionTerritory']==i]['Confirmed'], name=i,
                         line=dict(width=2)))

fig1.update_layout(
    title='Corona Virus Confirmed Cases Trend in Different States of India',
     yaxis=dict(
        title='Number of Cases Per Day')
    )

fig1.show()
states = covid_indStates['State/UnionTerritory'].unique()
fig1 = go.Figure()
for i in states:
    fig1.add_trace(go.Scatter(x=covid_india.index,
                          y=covid_indStates[covid_indStates['State/UnionTerritory']==i]['Cured'], name=i,
                         line=dict(width=2)))
fig1.update_layout(
    title='Corona Virus Cured Cases Trend in Different States of India',
     yaxis=dict(
        title='Number of Cases Per Day')
    )

    
fig1.show()
states = covid_indStates['State/UnionTerritory'].unique()
fig1 = go.Figure()
for i in states:
    fig1.add_trace(go.Scatter(x=covid_india.index,
                          y=covid_indStates[covid_indStates['State/UnionTerritory']==i]['Deaths'], name=i,
                         line=dict(width=2)))

fig1.update_layout(
    title='Corona Virus Death Cases Trend in Different States of India',
     yaxis=dict(
        title='Number of Cases Per Day')
    )

fig1.show()
covid19Personal.sort_values('Total Confirmed cases',ascending=False)[:10].iplot(kind='bar',
                                                                               x='State',
                                                                               color = ['blue','green','red'],
                                                                               title='Top 10 States with Total Confirmed Cases',
                                                                               xTitle='States',
                                                                               yTitle = 'Cases Count')
covid19Personal.sort_values('Cured',ascending=False)[:10].iplot(kind='bar',
                                                                               x='State',
                                                                               color = ['blue','green','red'],
                                                                               title='Top 10 States with Total Cured Cases',
                                                                               xTitle='States',
                                                                               yTitle = 'Cases Count')
covid19Personal.sort_values('Death',ascending=False)[:10].iplot(kind='bar',
                                                                               x='State',
                                                                               color = ['blue','green','red'],
                                                                               title='Top 10 States with Total Death Cases',
                                                                               xTitle='States',
                                                                               yTitle = 'Cases Count')
from statsmodels.tsa.seasonal import seasonal_decompose
result1 = seasonal_decompose(covid_india['Confirmed'], model='multiplicative')
result2 = seasonal_decompose(covid_india['Recovered'], model='add')
result3 = seasonal_decompose(covid_india['Deaths'], model='add')

fig = go.Figure()
fig.add_trace(go.Scatter(x=covid_india.index, y=result1.trend, name='Confirmed',
                         line=dict(color='blue', width=4)))
fig.add_trace(go.Scatter(x=covid_india.index, y=result2.trend, name='Recovered',
                         line=dict(color='green', width=4)))
fig.add_trace(go.Scatter(x=covid_india.index, y=result3.trend, name='Deaths',
                         line=dict(color='firebrick', width=4)))

fig.update_layout(
    title='Trend Component in Data',
     yaxis=dict(
        title='Number of Cases Per Day')
    )

fig.show()
corr = covid_india[['Confirmed','Recovered','Deaths']].corr()
mask = np.triu(np.ones_like(corr,dtype = bool))

plt.figure(dpi=100)
plt.title('Correlation Analysis')
sns.heatmap(corr,mask=mask,annot=True,lw=1,linecolor='white',cmap='Reds')
plt.xticks(rotation=0)
plt.yticks(rotation = 0)
plt.show()
from statsmodels.tsa.stattools import grangercausalitytests
grangercausalitytests(covid_india[['Confirmed','Recovered']],maxlag=4);
grangercausalitytests(covid_india[['Confirmed','Deaths']],maxlag=4);
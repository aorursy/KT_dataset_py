import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

file="/kaggle/input/covid19/COVID-19.csv"
df=pd.read_csv(file,parse_dates=['Date'])

df.rename(columns={'Country/Region' : 'country'},inplace=True)
df['Active']=df['Confirmed']-df['Deaths']-df['Recovered']

df=df.drop(['Province/State'],axis=1)
df
active=df.groupby('Date')['Active'].sum().reset_index()
active
recovered=df.groupby('Date')['Recovered'].sum().reset_index()
recovered
deaths=df.groupby('Date')['Deaths'].sum().reset_index()
deaths
confirmed=df.groupby('Date')['Confirmed'].sum().reset_index()
confirmed

plt.figure(figsize=(15,10))
#plt.plot_date(active['Date'].dt.date,active['Active'],color='blue',label='Active')
#plt.plot_date(deaths['Date'].dt.date,deaths['Deaths'],color='red',label='Deaths')
#plt.plot_date(confirmed['Date'].dt.date,confirmed['Confirmed'],color='orange',label='Confirmed')
#plt.plot_date(recovered['Date'].dt.date,recovered['Recovered'],color='green',label='Recovered')
#plt.legend(loc=2)
#plt.show()
fig=go.Figure(data=(go.Scatter(x=active['Date'],y=active['Active'],mode="lines+markers",name='Active'),
                    go.Scatter(x=deaths['Date'],y=deaths['Deaths'],mode="lines+markers",name='Deaths'),
                    go.Scatter(x=confirmed['Date'],y=confirmed['Confirmed'],mode="lines+markers",name='Confirmed'),
                    go.Scatter(x=recovered['Date'],y=recovered['Recovered'],mode="lines+markers",name='Recovered')))
fig.update_layout(title="COVID-19 TRENDS",xaxis_title="Date",yaxis_title="Cases")
fig.show()
world=df[df['Date']=='2020-05-02']
world
fig=px.choropleth(world,locations='country',locationmode='country names',color='Active',range_color=[1,10000],
                  hover_name='country',color_continuous_scale='Viridis',title='COVID-19 ACTIVE CASES-MAY-02',hover_data=world)
fig.show()
fig=px.choropleth(world,locations='country',locationmode='country names',color='Deaths',range_color=[1,10000],
                  hover_name='country',color_continuous_scale='Viridis',title='COVID-19 DEATH CASES-MAY-02',hover_data=world)
fig.show()
fig=px.choropleth(world,locations='country',locationmode='country names',color='Recovered',range_color=[1,10000],
                  hover_name='country',color_continuous_scale='Viridis',title='COVID-19 RECOVERED CASES-MAY-02',hover_data=world)
fig.show()
df.dtypes
top=world[world['Confirmed']>20000]

plt.figure(figsize=(12,10))
sns.barplot(top['Confirmed'],top['country'],color='orange')
sns.barplot(top['Active'],top['country'],color='red')
sns.barplot(top['Deaths'],top['country'],color='black')
plt.show()
from fbprophet import Prophet as PH
cmodel=PH(daily_seasonality=True)
amodel=PH(daily_seasonality=True)
rmodel=PH(daily_seasonality=True)
dmodel=PH(daily_seasonality=True)
confirmedpred=confirmed.rename(columns={'Date':'ds','Confirmed':'y'})
activepred=active.rename(columns={'Date':'ds','Active':'y'})
recoveredpred=recovered.rename(columns={'Date':'ds','Recovered':'y'})
deathpred=deaths.rename(columns={'Date':'ds','Deaths':'y'})
cmodel.fit(confirmedpred)
amodel.fit(activepred)
rmodel.fit(recoveredpred)
dmodel.fit(deathpred)
predDateC=cmodel.make_future_dataframe(periods=7)
predConfirmed=cmodel.predict(predDateC)
predDateA=amodel.make_future_dataframe(periods=7)
predActive=amodel.predict(predDateA)
predDateR=rmodel.make_future_dataframe(periods=7)
predRecovered=rmodel.predict(predDateR)
predDateD=dmodel.make_future_dataframe(periods=7)
predDeaths=dmodel.predict(predDateD)
fig=go.Figure(data=(go.Scatter(x=confirmed['Date'],y=confirmed['Confirmed'],mode="markers",name='CONFIRMED-TREND'),
              go.Scatter(x=predConfirmed['ds'],y=predConfirmed['trend'],mode='lines',name='CONFIRMED-PREDICTION'),
              go.Scatter(x=active['Date'],y=active['Active'],mode="markers",name='ACTIVE-TREND'),
              go.Scatter(x=predActive['ds'],y=predActive['trend'],mode='lines',name='ACTIVE-PREDICTION'),
              go.Scatter(x=recovered['Date'],y=recovered['Recovered'],mode="markers",name='RECOVERED-TREND'),
              go.Scatter(x=predRecovered['ds'],y=predRecovered['trend'],mode='lines',name='RECOVERED-PREDICTION'),
              go.Scatter(x=deaths['Date'],y=deaths['Deaths'],mode="markers",name='DEATHS-TREND'),
              go.Scatter(x=predDeaths['ds'],y=predDeaths['trend'],mode='lines',name='DEATHS-PREDICTION')))
fig.update_layout(title="COVID-19 ACTIVE-CASES(TREND VS PREDICTION)",xaxis_title="Date",yaxis_title="Cases")
fig.show()
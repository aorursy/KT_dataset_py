
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as plt
import datetime as dt
from collections import OrderedDict
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pandas.tseries.offsets import DateOffset


raw_data=pd.read_csv('../input/nepal_raw_data.csv')
df_worldWide=pd.read_csv('../input/owid-covid-data.csv')
recovered_df=pd.read_csv('../input/nepal_recovered.csv')
covid_df_nepal=df_worldWide[df_worldWide['location']=='Nepal']
covid_df_nepal=covid_df_nepal[['date','total_tests','total_deaths']]
columns=['date','case_number','age','gender','District','Province','total_case']
raw_data.drop(raw_data.columns.difference(
    ['Date of diagnosis**','Case No.','Age','Sex','Address/location','Province','Cumulative total']),1,inplace=True)
raw_data.columns=columns
raw_data.District.fillna(raw_data.Province,inplace=True)
raw_data.date=pd.to_datetime(raw_data.date).dt.strftime('%Y-%m-%d')
raw_data.case_number=raw_data.case_number.str.split(' ',expand=True)[1]
processed_data_district=raw_data.groupby(by=['date','District'],as_index=False).agg(OrderedDict([('case_number','count'),                                                                         ('Province','last')]))
#processed_data_district=raw_data[raw_data['Cumulative total']>1]
processed_data_province=processed_data_district.groupby(by=['date','Province'],as_index=False).sum()

raw_data=raw_data[raw_data['total_case']>1]
date_wise_case=raw_data.groupby(by='date',as_index=False).agg({'total_case':'last'})
date_wise_data=date_wise_case.merge(covid_df_nepal,on=['date'],how='left')
recovered_df.date=pd.to_datetime(recovered_df.date).dt.strftime('%Y-%m-%d')
date_wise_data=date_wise_data.merge(recovered_df,on=['date'],how='left')
#will use for prediction
data_for_prediction=date_wise_data
date_wise_data=date_wise_data[:-1]

date_wise_data['date']=pd.to_datetime(date_wise_data["date"])
weeks=date_wise_data["date"].dt.weekofyear
date_wise_data.insert(loc=1,column='week_of_year',value=weeks)

date_wise_data['new_case']=date_wise_data['total_case'].diff()
date_wise_data.iloc[0,date_wise_data.columns.get_loc('new_case')]=2

date_wise_data['new_test']=date_wise_data['total_tests'].diff()
date_wise_data.iloc[0,date_wise_data.columns.get_loc('new_test')]=1

date_wise_data['new_death']=date_wise_data['total_deaths'].diff()
date_wise_data.iloc[0,date_wise_data.columns.get_loc('new_death')]=1

date_wise_data['new_recovered']=date_wise_data['total_recovered'].diff()
date_wise_data.iloc[0,date_wise_data.columns.get_loc('new_recovered')]=1

date_wise_data['new_case']=date_wise_data['new_case'].astype(int)
date_wise_data['new_test']=date_wise_data['new_test'].astype(int)
date_wise_data['new_death']=date_wise_data['new_death'].astype(int)
date_wise_data['new_recovered']=date_wise_data['new_recovered'].astype(int)

date_wise_data[['growth_factor_confirmed','growth_factor_recovered','growth_factor_deaths']]=date_wise_data[['total_case','total_recovered','total_deaths']].pct_change().add(1)


fig1=go.Figure()
fig1.add_trace(go.Bar(x=date_wise_data.date,y=date_wise_data.total_case,name="Confirmed Cases Bar",
                     marker=dict(color="#cfcedb")))
fig1.add_trace(go.Scatter(x=date_wise_data.date,y=date_wise_data.total_case,mode='lines+markers',
                          line=dict(color="#003600",width=2),
                          name="Confirmed Case Line"))
fig1.add_trace(go.Scatter(x=date_wise_data.date,y=date_wise_data.total_case-date_wise_data.total_deaths-date_wise_data.total_recovered,
                          mode='lines+markers',
                           line=dict(color="#c41696",width=2),
                           marker=dict(color="#100d3b"),
                          name="Total Active Case"))
fig1.add_trace(go.Scatter(x=date_wise_data.date,y=date_wise_data.total_deaths,
                          mode='lines+markers',name="Death Cases"))
fig1.add_trace(go.Scatter(x=date_wise_data.date,y=date_wise_data.total_recovered,
                          line=dict(color="#007600",width=2),
                          mode='markers+lines',name="Recovered Cases"))
fig1.update_layout(title="Growth of Different Types of Cases in Nepal",xaxis_title="Date",yaxis_title="Number Of Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig1.show() 
#Graph For Growth Factor Daily
fig1=go.Figure()
x_data=date_wise_data.date
fig1.add_trace(go.Scatter(x=x_data,y=date_wise_data.growth_factor_confirmed,mode='lines+markers',
                          line=dict(color="#003600",width=2),
                          name="Confirmed Case"))

fig1.add_trace(go.Scatter(x=x_data,y=date_wise_data.growth_factor_deaths,
                          mode='lines+markers',name="Death Cases"))
fig1.add_trace(go.Scatter(x=x_data,y=date_wise_data.growth_factor_recovered,
                          line=dict(color="#007600",width=2),
                          mode='markers+lines',name="Recovered Cases"))
fig1.update_layout(title="Growth Factor of Different Cases",xaxis_title="Date",yaxis_title="Growth Factor",legend=dict(x=0,y=1,traceorder="normal"))
fig1.show() 
#Graph For New Case Daily
fig1=go.Figure()
x_data=date_wise_data.date
fig1.add_trace(go.Scatter(x=x_data,y=date_wise_data.new_case,mode='lines+markers',
                          line=dict(color="#003600",width=2),
                          name="Confirmed Case Line"))

fig1.add_trace(go.Scatter(x=x_data,y=date_wise_data.new_death,
                          mode='lines+markers',name="Death Cases"))
fig1.add_trace(go.Scatter(x=x_data,y=date_wise_data.new_recovered,
                          line=dict(color="#007600",width=2),
                          mode='markers+lines',name="Recovered Cases"))
fig1.update_layout(title="Daily New Different Cases in Nepal",xaxis_title="Date",yaxis_title="Number Of Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig1.show()  
#Daily Test vs Positive Cases
fig=go.Figure()
fig1=go.Figure()
fig=make_subplots(rows=1,cols=2,
                 subplot_titles=('Each Day\'s Tests','Each Day\'s Positive Cases'))
x_data=date_wise_data.date
fig.add_trace(go.Bar(x=x_data,y=date_wise_data.new_test,
                          name="Total Tests"),row=1,col=1,)

fig.add_trace(go.Bar(x=x_data,y=date_wise_data.new_case,
                         name="Positive Tests"),row=1,col=2,)
fig1.add_trace(go.Scatter(x=x_data,y=date_wise_data.new_case/date_wise_data.new_test))
fig.update_layout(title="Tests And Positive Tests in Each Day",xaxis_title="Date after 2nd Case seen ",yaxis_title="Number Of Tests/Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig1.update_layout(title="Positive Case per Test in Each Day",xaxis_title="Date after 2nd Case seen ",yaxis_title="Ratio",legend=dict(x=0.5,y=1.2,traceorder="normal"))
fig.show()
fig1.show()
#pre-process for weekly
covid_nepal_df_weekly=date_wise_data.sort_values('date').groupby(by='week_of_year',as_index=False).agg(
    OrderedDict([('date','last'),
                 ('total_case','last'),
                 ('new_case','sum'),
                 ('total_deaths','last'),
                 ('new_death','sum'),
                 ('total_tests','last'),
                 ('new_test','sum'),
                 ('total_recovered','last'),
                 ('new_recovered','sum'),
    ]))
covid_nepal_df_weekly.index = np.arange(1, len(covid_nepal_df_weekly) + 1)
fig1=go.Figure()
x_data=covid_nepal_df_weekly.index
fig1.add_trace(go.Scatter(x=x_data,y=covid_nepal_df_weekly.total_case,mode='lines+markers',
                          line=dict(color="#003600",width=2),
                          name="Total Confirmed Case"))
fig1.add_trace(go.Scatter(x=x_data,y=covid_nepal_df_weekly.total_case-covid_nepal_df_weekly.total_deaths-covid_nepal_df_weekly.total_recovered,
                          mode='lines+markers',
                           line=dict(color="#c41696",width=2),
                           marker=dict(color="#100d3b"),
                          name="Total Active Case"))
fig1.add_trace(go.Scatter(x=x_data,y=covid_nepal_df_weekly.total_deaths,
                          mode='lines+markers',name="Death Cases"))
fig1.add_trace(go.Scatter(x=x_data,y=covid_nepal_df_weekly.total_recovered,
                          line=dict(color="#007600",width=2),
                          mode='markers+lines',name="Recovered Cases"))
fig1.update_layout(title="Weekly Growth of Different Types of Cases in Nepal",xaxis_title="WeeK (After 2nd Case Seen)",yaxis_title="Number Of Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig1.show() 
fig1=go.Figure()
x_data=covid_nepal_df_weekly.index
fig1.add_trace(go.Scatter(x=x_data,y=covid_nepal_df_weekly.new_case,mode='lines+markers',
                          line=dict(color="#003600",width=2),
                          name="Confirmed Case Line"))

fig1.add_trace(go.Scatter(x=x_data,y=covid_nepal_df_weekly.new_death,
                          mode='lines+markers',name="Death Cases"))
fig1.add_trace(go.Scatter(x=x_data,y=covid_nepal_df_weekly.new_recovered,
                          line=dict(color="#007600",width=2),
                          mode='markers+lines',name="Recovered Cases"))
fig1.update_layout(title="Weekly New Cases in Nepal",xaxis_title="Date",yaxis_title="Number Of Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig1.show() 
fig=go.Figure()
fig1=go.Figure()
fig=make_subplots(rows=1,cols=2,
                 subplot_titles=('Each Week\'s Tests','Each Week\'s Positive Cases'))
x_data=covid_nepal_df_weekly.index
fig.add_trace(go.Bar(x=x_data,y=covid_nepal_df_weekly.new_test,
                          name="Total Tests"),row=1,col=1,)

fig.add_trace(go.Bar(x=x_data,y=covid_nepal_df_weekly.new_case,
                         name="Positive Tests"),row=1,col=2,)
fig1.add_trace(go.Scatter(x=x_data,y=covid_nepal_df_weekly.new_case/covid_nepal_df_weekly.new_test))
fig.update_layout(title="Tests And Positive Tests in each week",xaxis_title="Number Of Weeks after 2nd Case seen ",yaxis_title="Number Of Tests/Cases",legend=dict(x=0.5,y=1.2,traceorder="normal"))
fig1.update_layout(title="Positive Case per Test in each week",xaxis_title="Number Of Weeks after 2nd Case seen ",yaxis_title="Ratio",legend=dict(x=0.5,y=1.2,traceorder="normal"))
fig.show()
fig1.show() 
trace=[]
processed_data_province=processed_data_province[1:]
for i,x in enumerate(processed_data_province.Province.unique()):
    prov_data_each=processed_data_province[processed_data_province['Province']==x].sort_values('date')
    prov_data_each.case_number=prov_data_each.case_number.cumsum()
    trace.append(go.Scatter(x=prov_data_each.date,y=prov_data_each.case_number,name=x))
fig=go.Figure(data=trace)
fig.update_layout(title="Growth of Different Types of Cases in Different Province",xaxis_title="Date",yaxis_title="Number Of Cases",legend=dict(x=0,y=1,traceorder="normal"))

data_for_prediction=data_for_prediction[['date','total_case']]
data_for_prediction['new_cases']=data_for_prediction['total_case'].diff()
data_for_prediction=data_for_prediction.fillna(1)
data_for_prediction.new_cases=data_for_prediction.new_cases.astype('int')
data_for_prediction.date=pd.to_datetime(data_for_prediction.date)
data_for_prediction.set_index('date',inplace=True)
data_for_prediction.drop('total_case',axis=1,inplace=True)
# train=data_for_prediction
# scaler=MinMaxScaler()
# scaler.fit(train)
# train=scaler.transform(train)

# n_input=7
# n_feature=1
# generator=TimeseriesGenerator(train,train,length=n_input,batch_size=4)
# model=Sequential()
# model.add(LSTM(200,activation='relu',input_shape=(n_input,n_feature)))
# model.add(Dropout(0.15))
# model.add(Dense(1))
# model.compile(optimizer='adam',loss='mse')
# model.fit_generator(generator,epochs=1000)
pred_list=[]
batch=train[-n_input:].reshape((1,n_input,n_feature))
for i in range(n_input):
    pred_list.append(model.predict(batch)[0])
    batch=np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)
add_dates=[data_for_prediction.index[-1]+DateOffset(days=x) for x in range(0,8)]
future_dates=pd.DataFrame(index=add_dates[1:],columns=data_for_prediction.columns)
df_predict=pd.DataFrame(scaler.inverse_transform(pred_list),index=future_dates[-n_input:].index,columns=['Prediction'])
df_final=pd.concat([data_for_prediction,df_predict],axis=1)
df_final=df_final.reset_index().rename(columns={'index':'date'})
#prediction for 1 Week
fig1=go.Figure()

fig1.add_trace(go.Scatter(x=df_final.date,y=df_final.new_cases.cumsum(),mode='lines+markers',
                          line=dict(color="#003600",width=2),
                          name="Confirmed Case Line"))
fig1.add_trace(go.Scatter(x=df_final.date,y=df_final.Prediction.cumsum()+df_final.new_cases.sum(),
                          mode='markers',name="Predictions"))

fig1.update_layout(title="Growth of Confirmed Cases and Prediction for Next Week",xaxis_title="Date",yaxis_title="Number Of Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig1.show() 
df_final.tail(7)

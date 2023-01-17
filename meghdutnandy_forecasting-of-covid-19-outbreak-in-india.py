#Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import plotly.offline as py
import plotly.express as px
Cases=pd.read_csv('../input/covid_19_india.csv',index_col=[1])
Cases.tail(4)
Cases.index=pd.to_datetime(Cases.index,dayfirst=True)
Cases.drop(['Sno','Time',],axis='columns',inplace=True)
Cases.rename({'State/UnionTerritory':'State/UTs','Cured':'Recovered'},axis='columns',inplace=True)
Cases.tail(5)
Cases['ActiveCases']=Cases['Confirmed']-(Cases['Recovered']+Cases['Deaths'])
Cases.drop({'ConfirmedIndianNational','ConfirmedForeignNational'},axis='columns',inplace=True)
Cases.tail(4)
print('Earlist Entry :',Cases.index.min())
print('Last Entry    :',Cases.index.max())
print('Total Day     :',Cases.index.max()-Cases.index.min())
data_today=Cases[Cases.index=='2020-04-28']
data_today
fig = px.pie(data_today[data_today["Confirmed"]>100], values="Confirmed", names="State/UTs", title="Number of confirmed Cases by State/UT with major infection", template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()
fig = px.pie(data_today[data_today["Confirmed"]>100], values="Deaths", names="State/UTs", title="Number of Deaths by State/UT with major infection", template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()
fig = px.pie(data_today[data_today["Confirmed"]>100], values="Recovered", names="State/UTs", title="Number of Recovered by State/UT with major infection", template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()
date_wise = Cases.groupby(['Date','State/UTs','Confirmed'])['Recovered','Deaths','ActiveCases'].sum().reset_index().sort_values('Confirmed',ascending=False)
date_wise.head(4)
fig = px.bar(date_wise,height=500,x='Date',y='Confirmed',hover_data =['State/UTs','ActiveCases','Deaths'],color='Confirmed')
fig.show()
fig = px.bar(date_wise,height=500,x='Date',y='Deaths',hover_data =['State/UTs','ActiveCases','Deaths'],color='Confirmed')
fig.show()
Cases.plot()

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import matplotlib.ticker as tick

import plotly.express as px

import plotly.graph_objects as go

import datetime
#Reading live data from website 

states=pd.read_csv("https://api.covid19india.org/csv/latest/states.csv")
states.isnull().sum()

#Missing/Null data is only found in Tested column 
#Retrieving Latest data of all states and union territories

df=states.tail(36)

df.tail(5)

# Creating a data set without Cumilative Information of India to do State wise analysis

df=df[df.State != 'India']  

df.tail(5)
fig = px.bar(df, x='State', y='Confirmed',labels={

                     "Confirmed": "Confirmed Cases",

                     "State": " "})

fig.show()
piv=df.pivot_table(index=['State'],values=['Confirmed','Recovered'],aggfunc=sum)

fig=piv.plot(kind='bar',figsize=(30,10))

plt.show()

piv=df.pivot_table(index=['State'],values=['Confirmed','Deceased'],aggfunc=sum)

fig=piv.plot(kind='bar',figsize=(20,10))

plt.show()
#Finding the number of active cases and adding the column into data set

df['Active_Cases']=df['Confirmed']-df['Recovered']-df['Deceased']
plt.figure(figsize=(25,10))

chart=sns.barplot(x=df['State'],y=df['Active_Cases'])

chart.set_xticklabels(chart.get_xticklabels(), rotation=45,horizontalalignment='right',fontsize='x-large')

plt.show()

#Adding a month column to to perform monthwise analysis on data

states['Month'] = pd.DatetimeIndex(states['Date']).month

states['Active_Cases']=states['Confirmed']-states['Recovered']-states['Deceased']
df2=states[states.State == 'India']  

df2.tail(2)
df3=df2.groupby('Month').apply(np.mean)

df3.tail(2)
plt.plot(df3.Month,df3.Confirmed, color='blue', marker='o', linestyle='dashed', label='Confirmed')

plt.plot(df3.Month,df3.Recovered, color='violet', marker='o', linestyle='dotted', label='Recovered')

plt.legend()

plt.show()
plt.plot(df3.Month,df3.Active_Cases, color='green', marker='o', linestyle='solid', label='Active Cases')

plt.legend()

plt.show()

plt.plot(df3.Month,df3.Deceased, color='red', marker='o', linestyle='solid', label='Deceased')

plt.legend()

plt.show()
#Analysis can be done on any required state by simply changing the name of state in below line

df2=states[states.State == 'Kerala']  
#Finding mean of cases inorder to analyse Covid Cases

df3=df2.groupby('Month').apply(np.mean)

df3.tail(5)
plt.plot(df3.Month,df3.Confirmed, color='blue', marker='o', linestyle='dashed', label='Confirmed')

plt.plot(df3.Month,df3.Recovered, color='violet', marker='o', linestyle='dotted', label='Recovered')

plt.plot(df3.Month,df3.Active_Cases, color='green', marker='o', linestyle='dotted', label='Active Cases')

plt.legend()

plt.show()
fig = px.line(df3, x="Month", y="Deceased", title='Deceased')

fig.show()
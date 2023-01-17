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
my_path1="../input/covid19-in-india/covid_19_india.csv"
my_path2="../input/covid19-in-india/AgeGroupDetails.csv"
my_path3="../input/covid19-in-india/IndividualDetails.csv"
my_data1=pd.read_csv(my_path1)
my_data2=pd.read_csv(my_path2)
individual_details=pd.read_csv(my_path3)


my_data2
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize=(10,6))
sns.barplot(x=my_data2['AgeGroup'],y=my_data2['TotalCases'])
plt.show()
my_data1.tail()
#Number of missing data
my_data1.isna().sum()
my_data1['ActiveCases']=my_data1['Confirmed']-my_data1['Deaths']-my_data1['Cured']
from datetime import date
my_data1['Date'] = pd.to_datetime(my_data1['Date'])


#Temporary dataset for maximum date
#temp=my_data1.groupby('Date')['Confirmed','Deaths','Cured','ActiveCases'].sum().reset_index()
#temp=temp[temp['Date']==max(temp['Date'])].reset_index(drop=True)
#temp['Global Moratality'] = temp['Deaths']/temp['Confirmed']
#temp['Deaths per 100 Confirmed Cases'] = temp['Global Moratality']*100
#temp.head(20)
#import matplotlib.pyplot as plotter

#pieLabels = 'Deaths', 'Cured', 'ActiveCases'
#pieShare = [temp['Deaths'], temp['Cured'], temp['ActiveCases']]
#figureObject, axesObject = plotter.subplots()
#colors = ['yellowgreen', 'lightskyblue', 'lightcoral']


#Draw the pie chart-

#axesObject.pie(pieShare, labels=pieLabels, autopct='%1.2f', startangle=90,colors=colors)

#draw a circle at the center of pie to make it look like a donut
#(centre_circle = plt.Circle((0,0),0.50,color='black', fc='white',linewidth=1.25)
#fig = plt.gcf()
#fig.gca().add_artist(centre_circle)

#Aspect Ratio-i.e. pie chart is a circle
#axesObject.axis('equal')
#plotter.show()
state_datewise=my_data1.groupby('State/UnionTerritory')['Confirmed','ActiveCases','Deaths','Cured'].max()
state_datewise['ActiveCases']=state_datewise['Confirmed']-state_datewise['Deaths']-state_datewise['Cured']

state_datewise=state_datewise.reset_index().drop([14,24,32],axis=0)
state_datewise.reset_index()
statewise_cases=state_datewise.sort_values(by='Confirmed', ascending=False).reset_index()
statewise_cases=statewise_cases[['State/UnionTerritory','Confirmed','Deaths','Cured','ActiveCases']]
statewise_cases.style.background_gradient(cmap='Blues',subset=['Confirmed','ActiveCases'])\
                     .background_gradient(cmap='Reds',subset=['Deaths'])\
                     .background_gradient(cmap='Greens',subset=['Cured'])

import plotly.graph_objects as go

fig = go.Figure(data=[
    go.Bar(name='Active Cases', x=statewise_cases['State/UnionTerritory'], y=statewise_cases['Confirmed']),
    go.Bar(name='Deaths', x=statewise_cases['State/UnionTerritory'], y=statewise_cases['Deaths']),
    go.Bar(name='Recovered', x=statewise_cases['State/UnionTerritory'], y=statewise_cases['Cured'])
])
# Change the bar mode
fig.update_layout(barmode='group',height=700)
fig.update_yaxes(nticks=20,ticks="outside", tickwidth=2, tickcolor='crimson', ticklen=10)
fig.show()

India_confirmed=statewise_cases['Confirmed'].sum()
India_deaths=statewise_cases['Deaths'].sum()
India_cured=statewise_cases['Cured'].sum()
x=[India_confirmed,India_deaths,India_cured]
plt.figure(figsize=(8,8))
plt.title("Covid-19 Cases in India")
sns.barplot(x=['Total Cases','Deaths','Cured'],y=x)
sns.set_style("dark")
plt.show()
temp1=statewise_cases[statewise_cases['Deaths']>0][['State/UnionTerritory','Deaths','Confirmed']]
temp1.sort_values(by='Deaths',ascending=False)
temp1['DeathsPer100']=(temp1['Deaths']/temp1['Confirmed'])*100
temp1[['State/UnionTerritory','Deaths','DeathsPer100']].style.background_gradient(cmap='Reds')

individual_details
individual_details.isna().sum()
dummy=pd.get_dummies(individual_details['gender'])
pd.concat([individual_details,dummy],axis=1)
individual_details=individual_details.merge(dummy,left_index=True,right_index=True)
individual_details
female=individual_details['F'].sum()
male=individual_details['M'].sum()
total=female+male
female_percent=(female/total)*100
male_percent=(male/total)*100


import matplotlib.pyplot as plotter

pieLabels = 'Male', 'Female'
pieShare = [male, female]
figureObject, axesObject = plotter.subplots()
colors = ['yellowgreen', 'lightskyblue']


#Draw the pie chart-

axesObject.pie(pieShare, labels=pieLabels, autopct='%1.2f', startangle=90,colors=colors)

#draw a circle at the center of pie to make it look like a donut
centre_circle = plt.Circle((0,0),0.50,color='black', fc='white',linewidth=1.25)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

#Aspect Ratio-i.e. pie chart is a circle
axesObject.axis('equal')
plotter.show()

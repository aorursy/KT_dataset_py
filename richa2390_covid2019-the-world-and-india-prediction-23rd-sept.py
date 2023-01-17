import numpy as np # linear algebra import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

import os

# Any results you write to the current directory are saved as output.
%matplotlib inline 



import matplotlib

import matplotlib.pyplot as plt 

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
df= pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')

df.head()
df.shape
df.info()
df['ObservationDate'] = pd.to_datetime(df['ObservationDate'])

df['Last Update'] = pd.to_datetime(df['Last Update'])

df['Confirmed']=df['Confirmed'].astype('int')

df['Deaths']=df['Deaths'].astype('int')

df['Recovered']=df['Recovered'].astype('int')

from datetime import date

recent=df[['ObservationDate']][-1:].max()

df_update=df.loc[df.ObservationDate==pd.Timestamp(recent['ObservationDate'])]

#df_update=df.loc[df.ObservationDate<pd.Timestamp(date(2020,4,30))]

df_update
df_update.isnull().sum()
df_update['Province/State']=df_update.apply(lambda x: x['Country/Region'] if pd.isnull(x['Province/State']) else x['Province/State'],axis=1)

df['Province/State']=df.apply(lambda x: x['Country/Region'] if pd.isnull(x['Province/State']) else x['Province/State'],axis=1)
df_update['Country/Region']=df_update.apply(lambda x:'China' if x['Country/Region']=='Mainland China' else x['Country/Region'],axis=1)

df['Country/Region']=df.apply(lambda x:'China' if x['Country/Region']=='Mainland China' else x['Country/Region'],axis=1)
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

df_update['ProvinceID'] = le.fit_transform(df_update['Province/State'])

df_update['CountryID']=le.fit_transform(df_update['Country/Region'])

df_update.head()
corr= df_update.corr()

sns.heatmap(corr,annot=True)
print(df['Country/Region'].unique())

print("\nNumber of countries impacted by COVID 2019:",len(df['Country/Region'].unique()))
fig = plt.figure(figsize=(10,10))

conf_per_country = df_update.groupby('Country/Region')['Confirmed'].sum().sort_values(ascending=False)

conf_sum=df_update['Confirmed'].sum()

def absolute_value(val):

    a  = val

    return (np.round(a,2))

conf_per_country.plot(kind="pie",title='Percentage of confirmed cases per country',autopct=absolute_value)



plt.show ()
group_cases=df_update[['Confirmed','Recovered','Deaths','Country/Region']].groupby('Country/Region').sum().sort_values('Confirmed',ascending=False).head()

group_cases=group_cases.reset_index()

group_cases
group_cases['Recovery Rate']=round(group_cases['Recovered'] / group_cases['Confirmed'],2)

group_cases['Death Rate']=round(group_cases['Deaths'] / group_cases['Confirmed'],2)

group_cases=group_cases.sort_values(by='Confirmed', ascending= False)

group_cases.style.background_gradient(cmap='Greens')
f, ax = plt.subplots(figsize=(15, 10))

bar1=sns.barplot(x="Confirmed",y="Country/Region",data=group_cases,

            label="Confirmed", color="b")





bar2=sns.barplot(x="Recovered", y="Country/Region", data=group_cases,

            label="Recovered", color="g")





bar3=sns.barplot(x="Deaths", y="Country/Region", data=group_cases,

            label="Deaths", color="r")



ax.legend(loc=4, ncol = 1)

plt.xlabel("Confirmed Cases")

plt.show()
group_cases=df_update[['Recovered','Deaths','Country/Region']].groupby('Country/Region').sum().sort_values('Deaths',ascending=False).head(4)

group_cases.plot(kind='bar',width=0.5,colormap='plasma',figsize=(20,10))

plt.show()
china_cases=df_update.loc[df_update['Country/Region']=='China']
china_province_cases=china_cases[['Confirmed','Recovered','Deaths','Province/State']].groupby('Province/State').sum().sort_values('Confirmed',ascending=False)

china_province_cases.plot(kind='bar',width=0.95,colormap='rainbow',figsize=(20,6))

plt.show()
china_cases_complete=df.loc[df['Country/Region']=='China']

china_cases_complete['date'] = china_cases_complete['ObservationDate'].dt.date

china_cases_complete['date']=pd.to_datetime(china_cases_complete['date'])

china_cases_complete = china_cases_complete[china_cases_complete['date'] > pd.Timestamp(date(2020,1,21))]

num_plot = china_cases_complete.groupby('date')["Confirmed", "Recovered", "Deaths"].sum()

num_plot.plot(figsize=(8,8),title='Per Day statistics for China',colormap='rainbow',marker='o')

plt.show()
num_plot_china=num_plot.reset_index()

num_plot_china['Death Case Increase']=0

num_plot_china['Confirmed Case Increase']=0

num_plot_china['Confirmed Case Increase'][0]=0

num_plot_china['Death Case Increase'][0]=0

for i in range(1,num_plot_china.shape[0]):

    num_plot_china['Confirmed Case Increase'][i]=-(num_plot_china.iloc[i-1][1]-num_plot_china.iloc[i][1])

    num_plot_china['Death Case Increase'][i]=-(num_plot_china.iloc[i-1][3]-num_plot_china.iloc[i][3])

num_plot_china.tail()

    
num_plot_china['Confirmed Case Increase'].plot(kind='bar',width=0.95,figsize=(20,6),title='Confirmed Case Increase',colormap='rainbow')

plt.show()
num_plot_china['Death Case Increase'].plot(kind='bar',width=0.95,figsize=(20,6),title='Death Case Increase',colormap='rainbow')

plt.show()
hubei_cases_complete=df.loc[df['Province/State']=='Hubei']

hubei_cases_complete['date'] = hubei_cases_complete['ObservationDate'].dt.date

hubei_cases_complete['date']=pd.to_datetime(hubei_cases_complete['date'])

hubei_cases_complete = hubei_cases_complete[hubei_cases_complete['date'] > pd.Timestamp(date(2020,1,21))]

num_plot = hubei_cases_complete.groupby('date')["Confirmed", "Recovered", "Deaths"].sum()

num_plot.plot(figsize=(8,8),title='Per Day statistics for Hubei',colormap='rainbow',marker='o')

plt.show()
global_cases_complete=df.loc[~(df['Country/Region']=='China')]

global_cases_complete['date'] = global_cases_complete['ObservationDate'].dt.date

global_cases_complete['date']=pd.to_datetime(global_cases_complete['date'])

global_cases_complete = global_cases_complete[global_cases_complete['date'] > pd.Timestamp(date(2020,1,21))]

num_plot = global_cases_complete.groupby('date')["Confirmed", "Recovered", "Deaths"].sum()

num_plot.plot(figsize=(8,8),title='Per Day statistics for World',colormap='viridis',marker='o')

plt.show()
num_plot_global=num_plot.reset_index()

num_plot_global['Death Case Increase']=0

num_plot_global['Confirmed Case Increase']=0

num_plot_global['Confirmed Case Increase'][0]=0

num_plot_global['Death Case Increase'][0]=0

for i in range(1,num_plot_global.shape[0]):

    num_plot_global['Confirmed Case Increase'][i]=-(num_plot_global.iloc[i-1][1]-num_plot_global.iloc[i][1])

    num_plot_global['Death Case Increase'][i]=-(num_plot_global.iloc[i-1][3]-num_plot_global.iloc[i][3])

num_plot_global.tail(10)
num_plot_global['Confirmed Case Increase'].plot(kind='bar',width=0.95,figsize=(20,6),title='Confirmed Case Increase',colormap='viridis')

plt.show()
num_plot_global['Death Case Increase'].plot(kind='bar',width=0.95,figsize=(20,6),title='Death Case Increase',colormap='viridis')

plt.show()
italy_cases_complete=df.loc[df['Country/Region']=='Italy']

italy_cases_complete['date'] = italy_cases_complete['ObservationDate'].dt.date

italy_cases_complete['date']=pd.to_datetime(italy_cases_complete['date'])

italy_cases_complete = italy_cases_complete[italy_cases_complete['date'] > pd.Timestamp(date(2020,1,21))]

num_plot = italy_cases_complete.groupby('date')["Confirmed", "Recovered", "Deaths"].sum()

num_plot.plot(figsize=(8,8),title='Per Day statistics for Italy',marker='o')

plt.show()
num_plot_italy=num_plot.reset_index()

num_plot_italy['Death Case Increase']=0

num_plot_italy['Confirmed Case Increase']=0

num_plot_italy['Confirmed Case Increase'][0]=0

num_plot_italy['Death Case Increase'][0]=0

for i in range(1,num_plot_italy.shape[0]):

    num_plot_italy['Confirmed Case Increase'][i]=-(num_plot_italy.iloc[i-1][1]-num_plot_italy.iloc[i][1])

    num_plot_italy['Death Case Increase'][i]=-(num_plot_italy.iloc[i-1][3]-num_plot_italy.iloc[i][3])

num_plot_italy.tail()
num_plot_italy['Confirmed Case Increase'].plot(kind='bar',width=0.95,figsize=(20,6),title='Confirmed Case Increase')

plt.show()
num_plot_italy['Death Case Increase'].plot(kind='bar',width=0.95,figsize=(20,6),title='Death Case Increase')

plt.show()
us_cases_complete=df.loc[df['Country/Region']=='US']

us_cases_complete['date'] = us_cases_complete['ObservationDate'].dt.date

us_cases_complete['date']=pd.to_datetime(us_cases_complete['date'])

us_cases_complete = us_cases_complete[us_cases_complete['date'] > pd.Timestamp(date(2020,1,21))]

num_plot = us_cases_complete.groupby('date')["Confirmed", "Recovered", "Deaths"].sum()

num_plot.plot(figsize=(8,8),title='Per Day statistics for US',colormap='cool',marker='o')

plt.show()
num_plot_us=num_plot.reset_index()

num_plot_us['Death Case Increase']=0

num_plot_us['Confirmed Case Increase']=0

num_plot_us['Confirmed Case Increase'][0]=0

num_plot_us['Death Case Increase'][0]=0

for i in range(1,num_plot_us.shape[0]):

    num_plot_us['Confirmed Case Increase'][i]=-(num_plot_us.iloc[i-1][1]-num_plot_us.iloc[i][1])

    num_plot_us['Death Case Increase'][i]=-(num_plot_us.iloc[i-1][3]-num_plot_us.iloc[i][3])

num_plot_us.tail()
num_plot_us['Confirmed Case Increase'].plot(kind='bar',width=0.95,figsize=(20,6),colormap='cool',title='Confirmed Case Increase')

plt.show()
num_plot_us['Death Case Increase'].plot(kind='bar',width=0.95,figsize=(20,6),colormap='cool',title='Death Case Increase')

plt.show()
india_cases_complete=df.loc[df['Country/Region']=='India']

india_cases_complete['date'] = india_cases_complete['ObservationDate'].dt.date

india_cases_complete['date']=pd.to_datetime(india_cases_complete['date'])

india_cases_complete = india_cases_complete[india_cases_complete['date'] > pd.Timestamp(date(2020,1,21))]

num_plot = india_cases_complete.groupby('date')["Confirmed", "Recovered", "Deaths"].sum()

num_plot['Confirmed'][132]=287154

num_plot['Recovered'][132]=140928

num_plot['Deaths'][132]=8106

num_plot['Confirmed'][134]=309595

num_plot['Recovered'][134]=154235

num_plot['Deaths'][134]=8888

num_plot['Confirmed'][135]=321634

num_plot['Recovered'][135]=162327

num_plot['Deaths'][135]=9197

num_plot['Confirmed'][136]=333039

num_plot['Recovered'][136]=169684

num_plot['Deaths'][136]=9521

num_plot['Confirmed'][137]=343071

num_plot['Recovered'][137]=180324

num_plot['Deaths'][137]=9917

num_plot.plot(figsize=(8,8),colormap='winter',title='Per Day statistics for India',marker='o')

num_plot_india=num_plot.reset_index()

num_plot_india['Confirmed Case Increase']=0

num_plot_india['Death Case Increase']=0

#num_plot_india['Confirmed'][132]=287154

#num_plot_india['Recovered'][132]=140928

#num_plot_india['Deaths'][132]=8106

#num_plot_india['Confirmed'][134]=309595

#num_plot_india['Recovered'][134]=154235

#num_plot_india['Deaths'][134]=8888

#num_plot_india['Confirmed'][135]=321634

#num_plot_india['Recovered'][135]=162327

#num_plot_india['Deaths'][135]=9197

#num_plot_india['Confirmed'][136]=333039

#num_plot_india['Recovered'][136]=153792

#num_plot_india['Deaths'][136]=9521

#num_plot_india['Confirmed'][137]=343071

#num_plot_india['Recovered'][137]=180324

#num_plot_india['Deaths'][137]=9917



num_plot_india['Confirmed'][229]=4930236

num_plot_india['Recovered'][229]=3859399

num_plot_india['Deaths'][229]=80776



for i in range(1,num_plot_india.shape[0]):

    num_plot_india['Confirmed Case Increase'][i]=-(num_plot_india.iloc[i-1][1]-num_plot_india.iloc[i][1])

    num_plot_india['Death Case Increase'][i]=-(num_plot_india.iloc[i-1][3]-num_plot_india.iloc[i][3])

num_plot_india.drop(227,axis=0,inplace=True)

num_plot_india.tail(20)
num_plot_india['Confirmed Case Increase'].plot(kind='bar',width=0.95,colormap='winter',figsize=(20,6),title='Confirmed Case Increase')

plt.show()
num_plot_india['Death Case Increase'].plot(kind='bar',width=0.95,colormap='winter',figsize=(20,6),title='Death Case Increase')

plt.show()
from sklearn.preprocessing import LabelEncoder

from plotly.offline import iplot, init_notebook_mode

import math

import bokeh 

import matplotlib.pyplot as plt

import plotly.express as px

from urllib.request import urlopen

import json

from dateutil import parser

from bokeh.layouts import gridplot

from bokeh.plotting import figure, show, output_file

from bokeh.layouts import row, column

from bokeh.resources import INLINE

from bokeh.io import output_notebook

from bokeh.models import Span

import warnings

warnings.filterwarnings("ignore")

output_notebook(resources=INLINE)

le=LabelEncoder()



df.rename(columns={'Country/Region': 'Country', 'ObservationDate': 'Date'}, inplace=True)

df = df.fillna('unknown')

df['Country'] = df['Country'].str.replace('US','United States')

df['Country'] = df['Country'].str.replace('UK','United Kingdom') 

df['Country'] = df['Country'].str.replace('Mainland China','China')

df['Code']=le.fit_transform(df['Country'])

virus_data = df

#print(virus_data.head())

#print(len(virus_data))



top_country = virus_data.loc[virus_data['Date'] == virus_data['Date'].iloc[-1]]

top_country = top_country.groupby(['Code','Country'])['Confirmed'].sum().reset_index()

top_country = top_country.sort_values('Confirmed', ascending=False)

top_country = top_country[:100]

top_country_codes = top_country['Country']

top_country_codes = list(top_country_codes)

#print(top_country)



countries = virus_data[virus_data['Country'].isin(top_country_codes)]

countries_day = countries.groupby(['Date','Code','Country'])['Confirmed','Deaths','Recovered'].sum().reset_index()

#print(countries_day)





exponential_line_x = []

exponential_line_y = []

for i in range(16):

    exponential_line_x.append(i)

    exponential_line_y.append(i)



china = countries_day.loc[countries_day['Code']==43]



new_confirmed_cases_china = []

new_confirmed_cases_china.append( list(china['Confirmed'])[0] - list(china['Deaths'])[0] 

                           - list(china['Recovered'])[0] )



for i in range(1,len(china)):



    new_confirmed_cases_china.append( list(china['Confirmed'])[i] - 

                                     list(china['Deaths'])[i] - 

                                     list(china['Recovered'])[i])

    

    

italy = countries_day.loc[countries_day['Code']==102]



new_confirmed_cases_ita = []

new_confirmed_cases_ita.append( list(italy['Confirmed'])[0] - list(italy['Deaths'])[0] 

                           - list(italy['Recovered'])[0] )



for i in range(1,len(italy)):

    

    new_confirmed_cases_ita.append( list(italy['Confirmed'])[i] - 

                                  list(italy['Deaths'])[i] - 

                                  list(italy['Recovered'])[i])

    

    

skorea = countries_day.loc[countries_day['Code']==186]



new_confirmed_cases_skorea = []

new_confirmed_cases_skorea.append( list(skorea['Confirmed'])[0] - list(skorea['Deaths'])[0] 

                           - list(skorea['Recovered'])[0] )



for i in range(1,len(skorea)):

    

    new_confirmed_cases_skorea.append( list(skorea['Confirmed'])[i] - 

                                     list(skorea['Deaths'])[i] - 

                                    list(skorea['Recovered'])[i])

    

    

india = countries_day.loc[countries_day['Code']==96]



new_confirmed_cases_india = []

new_confirmed_cases_india.append( list(india['Confirmed'])[0] - list(india['Deaths'])[0] 

                           - list(india['Recovered'])[0] )



for i in range(1,len(india)):

    

    new_confirmed_cases_india.append( list(india['Confirmed'])[i] - 

                                     list(india['Deaths'])[i] - 

                                    list(india['Recovered'])[i])

    



spain = countries_day.loc[countries_day['Code']==188]



new_confirmed_cases_spain = []

new_confirmed_cases_spain.append( list(spain['Confirmed'])[0] - list(spain['Deaths'])[0] 

                           - list(spain['Recovered'])[0] )



for i in range(1,len(spain)):

    

    new_confirmed_cases_spain.append( list(spain['Confirmed'])[i] - 

                                     list(spain['Deaths'])[i] - 

                                    list(spain['Recovered'])[i])

    



us = countries_day.loc[countries_day['Code']==211]



new_confirmed_cases_us = []

new_confirmed_cases_us.append( list(us['Confirmed'])[0] - list(us['Deaths'])[0] 

                           - list(us['Recovered'])[0] )



for i in range(1,len(us)):

    

    new_confirmed_cases_us.append( list(us['Confirmed'])[i] - 

                                     list(us['Deaths'])[i] - 

                                    list(us['Recovered'])[i])

    

    

german = countries_day.loc[countries_day['Code']==77]



new_confirmed_cases_german = []

new_confirmed_cases_german.append( list(german['Confirmed'])[0] - list(german['Deaths'])[0] 

                           - list(german['Recovered'])[0] )



for i in range(1,len(german)):

    

    new_confirmed_cases_german.append( list(german['Confirmed'])[i] - 

                                     list(german['Deaths'])[i] - 

                                    list(german['Recovered'])[i])

    

p1=figure(plot_width=800, plot_height=550, title="COVID 2019 Trajectories for Countries")

p1.grid.grid_line_alpha=0.3

p1.xaxis.axis_label = 'Total number of Confirmed Cases (Log scale)'

p1.yaxis.axis_label = 'Total number of active cases (Log scale)'





p1.line(exponential_line_x, exponential_line_y, line_dash="4 4", line_width=1)



p1.line(np.log(list(china['Confirmed'])), np.log(new_confirmed_cases_china), color='red', 

        legend_label='China', line_width=3)

p1.circle(np.log(list(china['Confirmed'])[-1]), np.log(new_confirmed_cases_china[-1]), size=5)



p1.line(np.log(list(italy['Confirmed'])), np.log(new_confirmed_cases_ita), color='blue', 

        legend_label='Italy', line_width=3)

p1.circle(np.log(list(italy['Confirmed'])[-1]), np.log(new_confirmed_cases_ita[-1]), size=5)







p1.line(np.log(list(skorea['Confirmed'])), np.log(new_confirmed_cases_skorea), color='violet', 

        legend_label='South Korea', line_width=3)

p1.circle(np.log(list(skorea['Confirmed'])[-1]), np.log(new_confirmed_cases_skorea[-1]), size=5)





p1.line(np.log(list(india['Confirmed'])), np.log(new_confirmed_cases_india), color='orange', 

        legend_label='India', line_width=3)

p1.circle(np.log(list(india['Confirmed'])[-1]), np.log(new_confirmed_cases_india[-1]), size=5)



p1.line(np.log(list(spain['Confirmed'])), np.log(new_confirmed_cases_spain), color='brown', 

        legend_label='Spain', line_width=3)

p1.circle(np.log(list(spain['Confirmed'])[-1]), np.log(new_confirmed_cases_spain[-1]), size=5)



p1.line(np.log(list(us['Confirmed'])), np.log(new_confirmed_cases_us), color='green', 

        legend_label='United States', line_width=3)

p1.circle(np.log(list(us['Confirmed'])[-1]), np.log(new_confirmed_cases_us[-1]), size=5)



p1.line(np.log(list(german['Confirmed'])), np.log(new_confirmed_cases_german), color='black', 

        legend_label='Germany', line_width=3)

p1.circle(np.log(list(german['Confirmed'])[-1]), np.log(new_confirmed_cases_german[-1]), size=5)



p1.legend.location = "bottom_right"

#output_file("coronavirus.html", title="COVID2019 Trajectory")

show(p1)





import requests

import io

age_group = pd.read_csv('../input/covid19-in-india/AgeGroupDetails.csv')

india_covid = pd.read_csv('../input/covid19-in-india/covid_19_india.csv')

india_covid.to_csv('india_cases.csv',date_format='%Y-%m-%d')

india_covid_19=pd.read_csv('./india_cases.csv')

#india_covid_19 = pd.read_csv('../input/statewisetestingdetailsindiacsv/covid_19_india.csv',sep=',')

hospital_beds = pd.read_csv('../input/covid19-in-india/HospitalBedsIndia.csv')

individual_details = pd.read_csv('../input/covid19-in-india/IndividualDetails.csv')

#ICMR_details = pd.read_csv('../input/covid19-in-india/ICMRTestingDetails.csv')

ICMR_labs = pd.read_csv('../input/covid19-in-india/ICMRTestingLabs.csv')

state_testing = pd.read_csv('../input/statewisetestingdetailsindiacsv/statewise_tested_numbers_data.csv')

india_covid_19.tail()
#Removal of 'Unassigned' State/UnionTerritory

india_covid_19.rename(columns={'State/UnionTerritory': 'State', 'Cured': 'Recovered'}, inplace=True)

unassigned=india_covid_19[india_covid_19['State']=='Unassigned'].index

india_covid_19.drop(unassigned,axis=0,inplace=True)

unassigned1=india_covid_19[india_covid_19['State']=='Nagaland#'].index

india_covid_19.drop(unassigned1,axis=0,inplace=True)

unassigned2=india_covid_19[india_covid_19['State']=='Jharkhand#'].index

india_covid_19.drop(unassigned2,axis=0,inplace=True)

unassigned3=india_covid_19[india_covid_19['State']=='Madhya Pradesh#'].index

india_covid_19.drop(unassigned3,axis=0,inplace=True)

unassigned4=india_covid_19[india_covid_19['State']=='Cases being reassigned to states'].index

india_covid_19.drop(unassigned4,axis=0,inplace=True)

unassigned5=india_covid_19[india_covid_19['State']=='Telengana***'].index

india_covid_19.drop(unassigned5,axis=0,inplace=True)

unassigned6=india_covid_19[india_covid_19['State']=='Telangana***'].index

india_covid_19.drop(unassigned6,axis=0,inplace=True)

unassigned7=india_covid_19[india_covid_19['State']=='Telangana'].index

india_covid_19.drop(unassigned7,axis=0,inplace=True)

unassigned8=india_covid_19[india_covid_19['State']=='Dadar Nagar Haveli'].index

india_covid_19.drop(unassigned8,axis=0,inplace=True)

unassigned9=india_covid_19[india_covid_19['State']=='Daman & Diu'].index

india_covid_19.drop(unassigned9,axis=0,inplace=True)
#india_covid_19['Date'] = pd.to_datetime(india_covid_19['Date'],errors='coerce')

#statec=india_covid_19.loc[india_covid_19.Date<pd.Timestamp(date(2020,4,30))]

statewise_cases = pd.DataFrame(india_covid_19.groupby(['State'])['Confirmed', 'Deaths', 'Recovered'].max().reset_index())

statewise_cases["Country"] = "India" 

fig = px.treemap(statewise_cases, path=['Country','State'], values='Confirmed',color='Confirmed', hover_data=['State'])

fig.show()
labels = ['Male', 'Female']

sizes = []

sizes.append(list(individual_details['gender'].value_counts())[0])

sizes.append(list(individual_details['gender'].value_counts())[1])

explode = (0.05, 0)

colors = ['#ffcc99','#66b3ff']

plt.figure(figsize= (8,8))

plt.pie(sizes, explode=explode, labels=labels,colors=colors, autopct='%1.1f',startangle=90)

plt.title('Percentage of Gender (Ignoring the Missing Values)',fontsize = 10)

plt.show ()
fig = plt.figure(figsize=(10,10))

age_group=age_group.iloc[0:9,:]

age_dist_india = age_group.groupby('AgeGroup')['Sno'].sum().sort_values(ascending=False)

def absolute_value(val):

    a  = val

    return (np.round(a,2))

age_dist_india.plot(kind="pie",title='Case Distribution by Age (Ignoring Missing values)',autopct=absolute_value,colormap='Paired',startangle=90)



plt.show ()
india_covid_19['Deaths']=india_covid_19['Deaths'].replace(['0#','NaN'],0)

india_covid_19['Deaths']=india_covid_19['Deaths'].astype('int')
state_details = pd.pivot_table(india_covid_19, values=['Confirmed','Deaths','Recovered'], index='State', aggfunc='max')

state_details['Recovery Rate'] = round(state_details['Recovered'] / state_details['Confirmed'],2)

state_details['Death Rate'] = round(state_details['Deaths'] /state_details['Confirmed'], 2)

state_details = state_details.sort_values(by='Confirmed', ascending= False)

state_details.style.background_gradient(cmap='Purples')
testing=state_testing.groupby('State')['Total Tested'].max().sort_values(ascending=False).reset_index()

fig = px.bar(testing, 

             x="Total Tested",

             y="State", 

             orientation='h',

             height=800,

             title='Statewise Testing',

            color='State')

fig.show()
state_testing=state_testing.fillna(0)

state_test_details = pd.pivot_table(state_testing, values=['Total Tested','Positive','Negative'], index='State', aggfunc='max')

state_test_details['Positive Test Rate'] = round(state_test_details['Positive'] / state_test_details['Total Tested'],2)

#state_test_details['Negative Test Rate'] = round(state_test_details['Negative'] /state_test_details['Total Tested'])

state_test_details = state_test_details.sort_values(by='Total Tested', ascending= False)

state_test_details.style.background_gradient(cmap='Blues')
values = list(ICMR_labs['state'].value_counts())

states = list(ICMR_labs['state'].value_counts().index)

labs = pd.DataFrame(list(zip(values, states)), 

               columns =['values', 'states'])

fig = px.bar(labs, 

             x="values",

             y="states", 

             orientation='h',

             height=1000,

             title='Statewise Labs',

            color='states')

fig.show()
from plotly.subplots import make_subplots

import plotly.graph_objects as go

hospital_beds_states =hospital_beds.drop([36])

cols_object = list(hospital_beds_states.columns[2:8])

for cols in cols_object:

    hospital_beds_states[cols] = hospital_beds_states[cols].astype(int,errors = 'ignore')

top_5_primary = hospital_beds_states.nlargest(5,'NumPrimaryHealthCenters_HMIS')

top_5_community = hospital_beds_states.nlargest(5,'NumCommunityHealthCenters_HMIS')

top_5_district_hospitals = hospital_beds_states.nlargest(5,'NumDistrictHospitals_HMIS')

top_5_public_facility = hospital_beds_states.nlargest(5,'TotalPublicHealthFacilities_HMIS')

top_5_public_beds = hospital_beds_states.nlargest(5,'NumPublicBeds_HMIS')

top_rural_hos = hospital_beds_states.nlargest(5,'NumRuralHospitals_NHP18')

top_rural_beds = hospital_beds_states.nlargest(5,'NumRuralBeds_NHP18')

top_urban_hos = hospital_beds_states.nlargest(5,'NumUrbanHospitals_NHP18')

top_urban_beds = hospital_beds_states.nlargest(5,'NumUrbanBeds_NHP18')



plt.figure(figsize=(30,30))

plt.suptitle('Health Facilities in Top 5 States',fontsize=30)

plt.subplot(231)

plt.title('Primary Health Centers',fontsize=25)

plt.barh(top_5_primary['State/UT'],top_5_primary['NumPrimaryHealthCenters_HMIS'],color ='blue');



plt.subplot(232)

plt.title('Community Health Centers',fontsize=25)

plt.barh(top_5_community['State/UT'],top_5_community['NumCommunityHealthCenters_HMIS'],color = 'blue');



plt.subplot(233)

plt.title('Public Health Facilities',fontsize=25)

plt.barh(top_5_public_facility['State/UT'],top_5_public_facility['TotalPublicHealthFacilities_HMIS'],color='blue');



plt.subplot(234)

plt.title('District Hospitals',fontsize=25)

plt.barh(top_5_district_hospitals['State/UT'],top_5_district_hospitals['NumDistrictHospitals_HMIS'],color = 'orange');



plt.subplot(235)

plt.title('Rural Hospitals',fontsize=25)

plt.barh(top_rural_hos['State/UT'],top_rural_hos['NumRuralHospitals_NHP18'],color = 'orange');

plt.subplot(236)

plt.title('Urban Hospitals',fontsize=25)

plt.barh(top_urban_hos['State/UT'],top_urban_hos['NumUrbanHospitals_NHP18'],color = 'orange');

plt.tight_layout(rect=[0, 0.03, 1, 0.95])







plt.figure(figsize=(27,15))

plt.suptitle('Number of Beds in Top 5 States',fontsize=30);

plt.subplot(131)

plt.title('Rural Beds',fontsize=25)

plt.barh(top_rural_beds['State/UT'],top_rural_beds['NumRuralBeds_NHP18'],color = 'orange');



plt.subplot(132)

plt.title('Urban Beds',fontsize=25)

plt.barh(top_urban_beds['State/UT'],top_urban_beds['NumUrbanBeds_NHP18'],color = 'blue');

plt.subplot(133)

plt.title('Public Beds',fontsize=25)

plt.barh(top_5_public_beds['State/UT'],top_5_public_beds['NumPublicBeds_HMIS'],color = 'purple');

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#Current number of confirmed cases

ax = num_plot_india['Confirmed'].plot(title="Confirmed Cases in India",figsize=(8,8));

ax.set(xlabel="Date", ylabel="Confirmed Cases");
train = num_plot_india.iloc[:-3,:2]

test = num_plot_india.iloc[-3:,:2]

#num_plot_india.iloc[190:]
train.rename(columns={"date":"ds","Confirmed":"y"},inplace=True)

test.rename(columns={"date":"ds","Confirmed":"y"},inplace=True)

test = test.set_index("ds")

test = test['y']
from fbprophet import Prophet

pd.plotting.register_matplotlib_converters()

model = Prophet(changepoint_prior_scale=0.4, changepoints=['2020-04-09','2020-04-15','2020-04-24','2020-05-05','2020-05-17','2020-05-31','2020-06-18','2020-07-08','2020-07-28','2020-08-21','2020-09-20'])

model.fit(train)
future_dates = model.make_future_dataframe(periods=20)

forecast =  model.predict(future_dates)

ax = forecast.plot(x='ds',y='yhat',label='Predicted Confirmed Case',legend=True,figsize=(10,10))

test.plot(y='y',label='Actual Confirmed Cases',legend=True,ax=ax)
from fbprophet.diagnostics import performance_metrics

from fbprophet.diagnostics import cross_validation

df_cv = cross_validation(model, initial='90 days', period='40 days', horizon = '3 days')

df_cv.head()

df_p = performance_metrics(df_cv)

df_p.head()

forecast.tail(15)
from fbprophet import Prophet

model_india = Prophet(growth="logistic",changepoint_prior_scale=0.4,changepoints=['2020-04-09','2020-04-15','2020-04-24','2020-05-05','2020-05-17','2020-05-31','2020-06-18','2020-07-08','2020-07-28','2020-08-21','2020-09-20'])

pop = 1380004385 #from worldometers

train['cap'] = pop

model_india.fit(train)

# Future Prediction

future_dates = model_india.make_future_dataframe(periods=300)

future_dates['cap'] = pop

forecast =  model_india.predict(future_dates)

# Plotting

ax = forecast.plot(x='ds',y='yhat',label='Predicted Confirmed Cases',legend=True,figsize=(10,10))

test.plot(y='y',label='Actual Confirmed Counts',legend=True,ax=ax)

ax.set(xlabel="Date", ylabel="Confirmed Cases");
forecast.iloc[270:300]
from statsmodels.tsa.arima_model import ARIMA

import datetime

arima = ARIMA(train['y'], order=(3, 1, 0))

arima = arima.fit(trend='nc', full_output=True, disp=True)

forecast = arima.forecast(steps= 30)

pred = list(forecast[0])

start_date = train['ds'].max()

prediction_dates = []

for i in range(30):

    date = start_date + datetime.timedelta(days=1)

    prediction_dates.append(date)

    start_date = date

plt.figure(figsize= (20,10))

plt.xlabel("Dates",fontsize = 10)

plt.ylabel('Total cases',fontsize = 10)

plt.title("Predicted Values for the next 25 Days" , fontsize = 20)



plt.plot_date(y= pred,x= prediction_dates,linestyle ='dashed',color = 'blue',label = 'Predicted')

plt.plot_date(y=train['y'].tail(15),x=train['ds'].tail(15),linestyle = '-',color = 'orange',label = 'Actual')



pred=pd.DataFrame(forecast[0],columns=['Predicted'])

dates=pd.DataFrame(prediction_dates,columns=['Date'])

arima_df=pd.merge(dates,pred,right_index=True,left_index=True)

arima_df.tail(30)
test=test.reset_index()
df1=pd.DataFrame(forecast[0],columns=['yhat'])

df2=pd.DataFrame(prediction_dates,columns=['ds'])

df3=test['y']

df4=pd.merge(df2,df3,right_index=True,left_index=True)

df5=pd.merge(df4,df1,right_index=True,left_index=True)
df5['mse'],df5['rmse'],df5['mae'],df5['mape'],df5['mdape']=[0,0,0,0,0]
for t in range(len(test)):

    mape =  np.mean(np.abs(df5['yhat'][t] - df5['y'][t])/np.abs(df5['y'][t]))

    df5['mape'][t]="{:.5f}".format(mape)

    mdape =  np.median(np.abs(df5['yhat'][t] - df5['y'][t])/np.abs(df5['y'][t]))

    df5['mdape'][t]="{:.5f}".format(mdape)

    mae = np.mean(np.abs(df5['yhat'][t] - df5['y'][t]))

    df5['mae'][t]=mae

    mse = np.mean((df5['yhat'][t] - df5['y'][t])**2)

    df5['mse'][t]=mse

    rmse = np.mean((df5['yhat'][t] - df5['y'][t])**2)**.5

    df5['rmse'][t]=rmse
df5


num_plot = india_cases_complete.groupby('date')["Confirmed", "Recovered", "Deaths"].sum()

num_plot['Confirmed'][132]=287154

num_plot['Recovered'][132]=140928

num_plot['Deaths'][132]=8106

num_plot['Confirmed'][134]=309595

num_plot['Recovered'][134]=154235

num_plot['Deaths'][134]=8888

num_plot['Confirmed'][135]=321634

num_plot['Recovered'][135]=162327

num_plot['Deaths'][135]=9197

num_plot['Confirmed'][136]=333039

num_plot['Recovered'][136]=169684

num_plot['Deaths'][136]=9521

num_plot['Confirmed'][137]=343071

num_plot['Recovered'][137]=180324

num_plot['Deaths'][137]=9917

num_plot_india=num_plot.reset_index()

num_plot_india['Active']=0

for i in range(len(num_plot_india)):

    num_plot_india['Active'][i]=num_plot_india['Confirmed'][i]-num_plot_india['Recovered'][i]-num_plot_india['Deaths'][i]

num_plot_india
train_bed=pd.DataFrame(columns=['ds','y'])

test_bed=pd.DataFrame(columns=['ds','y'])

train_bed_y= num_plot_india.iloc[:-3,-1:]

train_bed_ds = num_plot_india.iloc[:-3,:1]

train_bed=pd.merge(train_bed_ds,train_bed_y,right_index=True,left_index=True)

train_bed.rename(columns={'date': 'ds', 'Active': 'y'}, inplace=True)

test_bed_y = num_plot_india.iloc[-3:,-1:]

test_bed_ds = num_plot_india.iloc[-3:,:1]

test_bed=pd.merge(test_bed_ds,test_bed_y,right_index=True,left_index=True)

test_bed.rename(columns={'date': 'ds', 'Active': 'y'}, inplace=True)
test_bed = test_bed.set_index("ds")

test_bed = test_bed['y']
num_bed=hospital_beds.iloc[36][7]+hospital_beds.iloc[36][9]+hospital_beds.iloc[36][11]

model_bed = Prophet(growth = "logistic",changepoints=['2020-04-09','2020-04-15','2020-04-24','2020-05-05','2020-05-17','2020-05-31','2020-06-18','2020-07-08','2020-07-28','2020-08-21','2020-09-20'])

bed_cap = num_bed 

train_bed['cap'] = bed_cap

model_bed.fit(train_bed)

# Future Prediction

future_dates = model_bed.make_future_dataframe(periods=200)

future_dates['cap'] = bed_cap

forecast =  model_bed.predict(future_dates)

# Plotting

ax = forecast.plot(x='ds',y='yhat',label='Predicted Active Cases',legend=True,figsize=(10,10))

test_bed.plot(y='y',label='Actual Active Counts',legend=True,ax=ax)

ax.set(xlabel="Date", ylabel="Active Cases");
forecast.iloc[230:250]
hospital_beds_active=hospital_beds

hospital_beds_active['Total_Beds']=0
for i in range(len(hospital_beds_active)):

    hospital_beds_active['Total_Beds'][i]=hospital_beds.iloc[i][7]+hospital_beds.iloc[i][9]+hospital_beds.iloc[i][11]
india_covid_19['Active'] = india_covid_19['Confirmed'] - india_covid_19['Deaths']- india_covid_19['Recovered']

india_state_cases=pd.DataFrame()

india_covid_19['Date'] = pd.to_datetime(india_covid_19['Date'],errors='coerce')

states =list(india_covid_19['State'].unique())

for state in states:

    mah_cases_complete=india_covid_19.loc[india_covid_19['State']==state]

    mah_cases_complete['Date'] = mah_cases_complete['Date'].dt.strftime("%Y-%m-%d")

    mah_cases_complete['Date']=pd.to_datetime(mah_cases_complete['Date'])

    mah_cases_complete = mah_cases_complete[(pd.Timestamp(datetime.date(2020,3,21)) < mah_cases_complete['Date']) & ( mah_cases_complete['Date']< pd.Timestamp(datetime.date(2020,9,23)))]

    india_state_cases=india_state_cases.append(mah_cases_complete)



india_state_cases.sample(10)



for state in states:

    beds_vs_active=india_state_cases.loc[india_state_cases['State']==state]

    active=india_state_cases.groupby(['State'])['Active'].max()

    beds=hospital_beds_active.groupby(['State/UT'])['Total_Beds'].max()

    beds_vs_active=pd.merge(active,beds,right_index=True,left_index=True)

beds_vs_active=beds_vs_active.reset_index()

bva=beds_vs_active[['Total_Beds','Active','index']].groupby('index').max().sort_values('Total_Beds',ascending=False).head(10)

bva.plot(kind='bar',width=0.6,colormap='bwr',figsize=(10,6))

plt.xlabel('States')

plt.show()
num_bed_mah=hospital_beds_active.loc[hospital_beds_active['State/UT']=='Maharashtra']['Total_Beds']

num_bed_mah=num_bed_mah.iloc[0]

num_mah=india_state_cases.loc[india_state_cases['State']=='Maharashtra'][['Date','Active']]

train_bed_mah=pd.DataFrame(columns=['ds','y'])

test_bed_mah=pd.DataFrame(columns=['ds','y'])

train_bed_mah_y= num_mah.iloc[:-3,-1:]

train_bed_mah_ds = num_mah.iloc[:-3,:1]

train_bed_mah=pd.merge(train_bed_mah_ds,train_bed_mah_y,right_index=True,left_index=True)

train_bed_mah.rename(columns={'Date': 'ds', 'Active': 'y'}, inplace=True)

test_bed_mah_y = num_mah.iloc[-3:,-1:]

test_bed_mah_ds = num_mah.iloc[-3:,:1]

test_bed_mah=pd.merge(test_bed_mah_ds,test_bed_mah_y,right_index=True,left_index=True)

test_bed_mah.rename(columns={'Date': 'ds', 'Active': 'y'}, inplace=True)
#num_mah.iloc[120:]
test_bed_mah = test_bed_mah.set_index("ds")

test_bed_mah = test_bed_mah['y']
model_bed_mah = Prophet(growth = "logistic",changepoints=['2020-04-14','2020-04-21','2020-04-30','2020-05-13','2020-05-27','2020-06-15','2020-07-15','2020-09-05'])

bed_cap = num_bed_mah+200000

train_bed_mah['cap'] = bed_cap

model_bed_mah.fit(train_bed_mah)

# Future Prediction

future_dates = model_bed_mah.make_future_dataframe(periods=200)

future_dates['cap'] = bed_cap

forecast_mah =  model_bed_mah.predict(future_dates)

# Plotting

ax = forecast_mah.plot(x='ds',y='yhat',label='Predicted Active Cases',legend=True,figsize=(8,8))

test_bed_mah.plot(y='y',label='Actual Active Counts',legend=True,ax=ax)

ax.set(xlabel="Date", ylabel="Active Cases in Maharashtra");
forecast_mah.iloc[190:250]
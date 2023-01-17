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
import matplotlib.pyplot as plt

import seaborn as sns

data_x = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')

data_x.head()
data_x.info()
data_x.isnull().sum()
data_x.shape
data_x.nunique()
data_x['Victims'] = data_x['Confirmed'] - data_x['Recovered'] - data_x['Deaths']

data_x['Victims'].head()
data_x['ObservationDate'] = pd.to_datetime(data_x['ObservationDate'])

data_x['ObservationDate'].head()




data_x['Days'] = data_x['ObservationDate'] - pd.to_datetime(['2020-01-22']*len(data_x))

data_x.head()
data_x["Days"] = data_x["Days"].astype('timedelta64[D]')

data_x.head()
sns.set(style='whitegrid')

sns.boxplot(data_x['Confirmed'],color='g')
chin = data_x[data_x['Country/Region']=='Mainland China']

chin.head()
plt.scatter(chin['Days'],chin['Deaths'])

plt.xlabel("Days")

plt.ylabel("Death_toll")

plt.title("Deaths In Mainland China")
# Let's Check for Italy and India.



ita = data_x[data_x['Country/Region']=='Italy']

plt.scatter(ita['Days'],ita['Deaths'])

plt.xlabel("Days")

plt.ylabel("Death_toll")

plt.title("Deaths In Italy")
plt.scatter(ita['Days'],ita['Victims'])

plt.xlabel("Days")

plt.ylabel("Victims_toll")

plt.title("Victims In Italy")
ita.describe()
ita.nunique()
ind = data_x[data_x['Country/Region']=='India']

plt.scatter(ind['Days'],ind['Victims'])

plt.xlabel("Days")

plt.ylabel("No. of Victims")

plt.title("India's Statistics Against Corona")
ind
ind.nunique()
x = data_x[data_x['Country/Region']=="Mainland China"]

x.head()
x.nunique()
sns.set(style='whitegrid')

sns.boxplot(x['Deaths'],color='b')
sns.set(style='whitegrid')

sns.boxplot(x['Confirmed'],color='b')
x.describe()
x.isnull().sum()
plt.figure(figsize=(10,5))

plt.grid(True)

plt.scatter(data_x['Days'],data_x['Deaths'],marker='o',color='r')

plt.legend(loc=2)

plt.xlabel('Number of Days')

plt.ylabel('Deaths')

plt.title('Total Death Count ')
plt.plot(x.Victims[:100],'b',label='Active Victims')

plt.plot(x.Deaths[:100],'r',label='Deaths')
sns.jointplot(x['Confirmed'],x['Deaths'], kind='scatter',height=6 ,color='skyblue' )

plt.title('Death v/s Confirmed Cases')
from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()



import plotly.graph_objs as go 
trace = go.Bar(x=x["Province/State"],y=x["Victims"])

data = [trace]

layout = {"title":"Victims in Different Provinces of China",

         "xaxis":{"title":"Provinces/States","tickangle":90},

         "yaxis":{"title":"Number of Victims"}}

fig = go.Figure(data = data,layout=layout)

iplot(fig)
trace = go.Bar(x=x["Province/State"],y=x["Deaths"])

data = [trace]

layout = {"title":"Deaths in Different Provinces of China",

         "xaxis":{"title":"Provinces/States","tickangle":90},

         "yaxis":{"title":"Number of Deaths"}}

fig = go.Figure(data = data,layout=layout)

iplot(fig)
trace = go.Bar(x=x["Days"],y=x["Deaths"])

data = [trace]

layout = {"title":"Deaths in China with respect to Days",

         "xaxis":{"title":"No. of Days","tickangle":0},

         "yaxis":{"title":"Number of Deaths"}}

fig = go.Figure(data = data,layout=layout)

iplot(fig)
## Let's see the Recovery Rate of China:

trace = go.Bar(x=x["Days"],y=x["Recovered"])

data = [trace]

layout = {"title":"Recovery in China",

         "xaxis":{"title":"Days","tickangle":0},

         "yaxis":{"title":"Recovered Total"}}

fig = go.Figure(data = data,layout=layout)

iplot(fig)
x.head()
## Let's try to make a dataset with Countries, Deaths and Victims.



Total_days = len(x['Days'].value_counts())





C = pd.DataFrame(x.groupby('Days')['Deaths','Victims','Confirmed','Recovered'].sum())

C['Days'] = C.index

C.index=np.arange(1, Total_days + 1)



stat = C[['Days','Deaths','Victims','Confirmed','Recovered']]

stat.sort_values(by=['Days'],ascending=True)
tr1 = go.Bar(x=stat['Days'],y=stat["Confirmed"],name="Confirmed Cases")

tr = go.Bar(x=stat['Days'],y=stat["Recovered"],name="Recovered Patients")

layout = {'title':'Confirmed Cases v/s Recovered Patients in China',

         "xaxis":{"title":"No. of Days"},

         "yaxis":{"title":"No. of Patients"},

         "barmode":"group"}

data = [tr1,tr]



fig=go.Figure(data=data,layout=layout)

iplot(fig)
tr1 = go.Bar(x=stat['Days'],y=stat["Confirmed"],name="Confirmed Cases")

tr = go.Bar(x=stat['Days'],y=stat["Recovered"],name="Recovered Patients")

layout = {'title':'Confirmed Cases v/s Recovered Patients',

         "xaxis":{"title":"No. of Days"},

         "yaxis":{"title":"No. of Patients"},

         "barmode":"stack"}

data = [tr,tr1]



fig=go.Figure(data=data,layout=layout)

iplot(fig)
# Let's try to visualize similar data for Italy as well.



trace = go.Bar(x=ita["Days"],y=ita["Deaths"])

data = [trace]

layout = {"title":"Deaths in Italy",

         "xaxis":{"title":"No.of Days","tickangle":0},

         "yaxis":{"title":"Number of Deaths"}}

fig = go.Figure(data = data,layout=layout)

iplot(fig)

# data.groupby('Customer_status')['ActiveSinceDays'].mean()      #customer status and ASDays mean

y = x.groupby('Province/State')['Deaths']

y.nunique()
h = x[x['Province/State']=='Hubei']

# h.drop(['Case','States'],axis=1,inplace= True)

h
trace = go.Bar(x=h["Days"],y=h["Deaths"])

data = [trace]

layout = {"title":"Deaths in Hubei(China)",

         "xaxis":{"title":"No.of Days","tickangle":0},

         "yaxis":{"title":"Number of Deaths"}}

fig = go.Figure(data = data,layout=layout)

iplot(fig)

trace = go.Bar(x=h["Days"],y=h["Recovered"],name="Recovered")

trace1 = go.Bar(x=h["Days"],y=h["Deaths"],name="Deaths")

data = [trace,trace1]

layout = {"title":"Recovery v/s Deaths in Hubei(China)",

         "xaxis":{"title":"No.of Days","tickangle":0},

         "yaxis":{"title":"Recovery and Death rate"},

         "barmode":"group"}

fig = go.Figure(data = data,layout=layout)

iplot(fig)

sns.boxplot(h['Deaths'])
## Let's try to make a dataset with Countries, Deaths and Victims.



Total_countries = len(data_x['Country/Region'].value_counts())





COR = pd.DataFrame(data_x.groupby('Country/Region')['Deaths','Victims'].sum())

COR['Country/Region'] = COR.index

COR.index=np.arange(1, Total_countries + 1)



new = COR[['Country/Region','Deaths','Victims']]

new.sort_values(by=['Deaths'],ascending=False)
import plotly.express as px



fig = px.bar(new[['Country/Region', 'Victims']].sort_values('Victims', ascending=False), 

             y="Victims", x="Country/Region", color='Country/Region', 

             log_y=True, template='ggplot2', title='VICTIMS ANALYSIS')

fig.show()



fig = px.bar(new[['Country/Region', 'Deaths']].sort_values('Deaths', ascending=False), 

             y="Deaths", x="Country/Region", color='Country/Region', title='Deaths',

             log_y=True, template='ggplot2')

fig.show()
# fig = px.pie(df, values='pop', names='country', title='Population of European continent')

# fig.show()





fig = px.pie(new , values='Deaths',names='Country/Region',title='Deaths Due to COVID')

fig.show()

fig = px.pie(new , values='Victims',names='Country/Region',title='Victims Analysis')

fig.show()
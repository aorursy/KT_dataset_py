# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib import pyplot

%matplotlib inline

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200131.csv", delimiter=',')

df.head()
df.info()
df = df.fillna({"Province/State": "Unknown"})

df = df.fillna(0)  
df.isnull().sum()
df['Province/State'] = df['Province/State'].astype('category')

df['Country/Region'] = df['Country/Region'].astype('category')

df = df.rename(columns={'Country/Region': 'Country', 'Province/State': 'Province'})

df.head()
df.Country.unique()
for i,col in enumerate(df.columns):

    print(i+1,". column: ",col)
china= df[df['Country'] == 'Mainland China']

china= pd.DataFrame(china.groupby(['Province'])['Confirmed','Suspected','Recovered','Death'].agg('sum')).reset_index()

china.head(35)
china.sort_values(by=['Confirmed'], inplace=True,ascending=False)



plt.figure(figsize=(25,12))



#  title

plt.title("Number of Provinces Confirmed People in China")



# Bar chart showing Number of Patients Confirmed Infected by Corona Virus, by Country

sns.barplot(y=china['Province'],x=china['Confirmed'],orient='h')





# Add label for vertical axis

plt.ylabel("Number of Confirmed People")
other= df[df['Country'] != 'Mainland China']

other= pd.DataFrame(other.groupby(['Province'])['Confirmed','Suspected','Recovered','Death'].agg('sum')).reset_index()

other.head(35)
other.sort_values(by=['Confirmed'], inplace=True,ascending=False)



plt.figure(figsize=(25,14))



#  title

plt.title("Number of Confirmed People, not induding China")



# Bar chart showing Number of Patients Confirmed Infected by Corona Virus, by Country

sns.barplot(y=other['Province'],x=other['Confirmed'],orient='h')





# Add label for vertical axis

plt.ylabel("Count of Confirmed People by Province/State")
other.sort_values(by=['Suspected'], inplace=True,ascending=False)



plt.figure(figsize=(25,10))



#  title

plt.title("Number of People Suspected, not in China")



# Bar chart showing Number of Patients Confirmed Infected by Corona Virus, by Country

sns.barplot(y=other['Province'],x=other['Suspected'],orient='h')





# Add label for vertical axis

plt.ylabel("Number of Suspected People")
other.sort_values(by=['Recovered'], inplace=True,ascending=False)



plt.figure(figsize=(25,10))



#  title

plt.title("Number of Recovered People, not in China")



# Bar chart showing Number of Patients Confirmed Infected by Corona Virus, by Country

sns.barplot(y=other['Province'],x=other['Recovered'],orient='h')





# Add label for vertical axis

plt.ylabel("Number of Recovered People")
plt.figure(figsize=(12,8))

sns.catplot(x = "Confirmed", y = "Country", kind = "bar", 

            height=5, # make the plot 5 units high

            aspect=3,

            palette = "bright", 

            #edgecolor = ".6", data = df) 

            data = df)

           #set_xticklabels=45) 

plt.show()
df.Province.value_counts()
plt.figure(figsize=(14,9))

sns.catplot(x = "Confirmed", y = "Province", kind = "bar", 

            height=8, # make the plot 5 units high

            aspect=2,

            palette = "muted", 

            #edgecolor = ".6", data = df) 

            data = df[:22],

            orient='h')

           #set_xticklabels=45) 

plt.show()
import plotly.graph_objs as go

import plotly.figure_factory as ff

from IPython.display import HTML, Image

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.offline as py

import plotly.express as px



fig = go.Figure(

iplot([go.Scatter(x=df['Country'], y= df['Confirmed'], mode= 'markers')])

)

iplot
fig = go.Figure(

    data=[go.Bar(x=df['Province'], y=df['Confirmed'])],

    layout=dict(title=dict(text="A Bar Chart Confirmed Provinces"))

)

fig.show()
df2 = df['Country'].value_counts()



iplot([go.Choropleth(

    locationmode='country names',

    locations=df2.index.values,

    text=df2.index,

    z=df2.values

)])
#df3 = df[df['Country'] != 'Mainland China']

df3 = df[df['Confirmed'] >= 1]

df3.head()
df3.Country.value_counts()
#import plotly.graph_objs as go

#import plotly.figure_factory as ff

#from IPython.display import HTML, Image

#from plotly.offline import init_notebook_mode, iplot

#init_notebook_mode(connected=True)

#import plotly.offline as py

#import plotly.express as px
#china = df[df['Country'] == 'Mainland China'] or df3 = df[df['Country'] != 'Mainland China']

#                                                 df3 = df[df['Confirmed'] > 1]

#other = df[df['Country'] != 'Mainland China'] 



trace = go.Pie(labels = ['china', 'other'], values = df['Country'].value_counts(), 

               textfont=dict(size=15), opacity = 0.8,

               marker=dict(colors=['lightskyblue','gold'], 

                           line=dict(color='#000000', width=1.5)))





layout = dict(title =  'Distribution of China vs Other Countries')

           

fig = dict(data = [trace], layout=layout)

py.iplot(fig)
hist_data = [df['Confirmed'],df['Suspected'],df['Recovered'],df['Death']]

group_labels = list(df.iloc[:,3:7].columns)



fig = ff.create_distplot(hist_data, group_labels, bin_size=5)

iplot(fig, filename='Distplot of all corona stats')
c = go.Box(y=df["Confirmed"],name="Confirmed")

s = go.Box(y=df["Suspected"],name="Suspected")

r = go.Box(y=df["Recovered"],name="Recovered")

d = go.Box(y=df["Death"],name="Death")

 

data = [c,s,r,d]

iplot(data)
fig = ff.create_scatterplotmatrix(df.iloc[:,3:7], index='Confirmed', diag='box', size=2, height=800, width=800)

iplot(fig, filename ='Scatterplotmatrix.png',image='png')
#china = df[df['Country/Region'] == 'Mainland China']

#other = df[df['Country/Region'] != 'Mainland China']



def count():

    trace = go.Bar( x = df['Country'].value_counts().values.tolist(), 

                    y = ['Confirmed', 'Suspected', 'Recovered', 'Deaths'], 

                    orientation = 'h', 

                    text=df['Country'].value_counts().values.tolist(), 

                    textfont=dict(size=15),

                    textposition = 'auto',

                    opacity = 0.8,marker=dict(

                    color=['lightskyblue', 'gold'],

                    line=dict(color='#000000',width=1.5)))



    layout = dict(title =  'Count of variables')



    fig = dict(data = [trace], layout=layout)

    py.iplot(fig)



def percent():

    trace = go.Pie(labels = ['Confirmed', 'Suspected', 'Recovered', 'Deaths'], values = df['Country'].value_counts(), 

                   textfont=dict(size=15), opacity = 0.8,

                   marker=dict(colors=['lightskyblue', 'gold'], 

                               line=dict(color='#000000', width=1.5)))





    layout = dict(title =  'Percentage of features: Confirmed, Suspected, Recovered, Death')



    fig = dict(data = [trace], layout=layout)

    py.iplot(fig)
count()

percent()
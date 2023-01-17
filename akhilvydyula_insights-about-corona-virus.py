# insights about corona virus

import warnings 



warnings.filterwarnings("ignore") 
!ls ../input/novel-corona-virus-2019-dataset/
#to read the data 2019_nCoV_data.csv

import numpy as np

import pandas as pd



data_corona = pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")

# display top five rows in dataset

data_corona.head()
#display last five rows in dataset

data_corona.tail()

data_corona['Date'] = data_corona['Date'].apply(pd.to_datetime).dt.normalize()

data_corona['Last Update'] = data_corona['Last Update'].apply(pd.to_datetime).dt.normalize()

data_corona.head()
#shape of the dataset

data_corona.shape
del data_corona['Sno']
#now lets see dataframe

data_corona.head()
data_corona.describe()
# basic visualize to help counts in each country

data_corona['Country'].value_counts()

#list of all countries that present in country column

countries = data_corona['Country'].unique()

print(countries)



print("\nTotal countries affected by virus: ",len(countries))
data_corona.groupby('Country')['Confirmed'].sum()

data_corona.groupby(['Country','Province/State']).sum().head(50)
data_corona.groupby(['Country','Date']).sum().head(50)
data_corona.groupby(['Country','Last Update']).sum().head(50)
%matplotlib inline

# Import dependencies

from datetime import datetime

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import numpy as np

import matplotlib.pyplot as plt



import seaborn as sns



import scipy.stats as stats

import plotly.express as px

import plotly.graph_objs as go

from plotly.offline import iplot, init_notebook_mode

import plotly.figure_factory as ff

from plotly import subplots

from plotly.subplots import make_subplots

import plotly.graph_objs as go

from datetime import date

from fbprophet import Prophet

import math
# Exploring word cloud based on STATE value

from wordcloud import WordCloud

plt.subplots(figsize=(25,15))

wordcloud = WordCloud(background_color='black',

                          width=1920,

                          height=1080

                         ).generate(" ".join(data_corona.Country))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('country.png')

plt.show()
import pandas_profiling

pandas_profiling.ProfileReport(data_corona)
data_corona.groupby('Date')['Confirmed','Recovered','Deaths'].sum()
#rate of confirmed cases and recovered cases



#(confirmed*100)/total no of cases



# Ploting daily updtes for 

fig_d = go.Figure()

fig_d.add_trace(go.Scatter(x=data_corona.Date, y=data_corona.Confirmed, mode="lines+markers", name=f"MAX. OF {int(data_corona.Confirmed.max()):,d}" + ' ' + "CONFIRMED",line_color='red'))

fig_d.add_trace(go.Scatter(x=data_corona.Date, y=data_corona.Recovered, mode="lines+markers", name=f"MAX. OF {int(data_corona.Recovered.max()):,d}" + ' ' + "RECOVERED",line_color='deepskyblue'))

fig_d.add_trace(go.Scatter(x=data_corona.Date, y=data_corona.Deaths, mode="lines+markers", name=f"MAX. OF {int(data_corona.Deaths.max()):,d}" + ' ' + "DEATHS",line_color='Orange'))

fig_d.update_layout(template="ggplot2",title_text = '<b>Daily numbers for Confirmed, Death and Recovered </b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'), showlegend=True)

fig_d.update_layout(

    legend=dict(

        x=0.01,

        y=.98,

        traceorder="normal",

        font=dict(

            family="sans-serif",

            size=7,

            color="Black"

        ),

        bgcolor="White",

        bordercolor="black",

        borderwidth=1

    ))

fig_d.show()

fig = px.bar(data_corona[['Country', 'Confirmed']].sort_values('Confirmed', ascending=True), 

             y="Confirmed", x="Country", color='Country', 

             log_y=True, template='ggplot2', title='Recovered Cases')

fig.show()
fig = px.bar(data_corona[['Country', 'Recovered']].sort_values('Recovered', ascending=True), 

             y="Recovered", x="Country", color='Country', 

             log_y=True, title='Confirmed Cases')

fig.show()
fig=px.bar(data_corona[['Country','Deaths']].sort_values('Deaths',ascending=True),

       x="Country",y="Deaths",color='Country',

       log_y=True,title='Death cases')



fig.show()
#looking fro lastest data world wide

!ls ../input/novel-corona-virus-2019-dataset
data_corona = data_corona[data_corona['Confirmed'] != 0]
plt.figure(figsize=(30,10))

sns.barplot(x='Country',y='Confirmed',data=data_corona,color='red')

plt.tight_layout()
plt.figure(figsize=(30,10))

sns.barplot(x='Country',y='Recovered',data=data_corona,color='blue')

plt.tight_layout()
plt.figure(figsize=(30,10))

sns.barplot(x='Country',y='Deaths',data=data_corona,color='pink')

plt.tight_layout()
data_corona.groupby("Country")["Deaths"].plot.bar()


g = sns.PairGrid(data_corona)

g.map(plt.scatter);
data_2 = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")



data_2.head(4)
data_2.describe()
import pandas_profiling

pandas_profiling.ProfileReport(data_2)
del data_2["SNo"]



data_2.head(3)
data_2.groupby('Province/State')['Deaths'].sum().reset_index().sort_values(by=['Deaths'],ascending=False).head()
data_corona.groupby('Province/State')['Deaths'].sum().reset_index().sort_values(by=['Deaths'],ascending=False).head()
# Reading Data



covid_open=pd.read_csv('../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv')

covid_confirmed=pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

covid_death= pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
print(covid_open.shape)

covid_open.describe()
print("The shapeof the data death cases ",covid_death.shape)

covid_death.describe()
print("The shapeof the data confirmed cases ",covid_confirmed.shape)

covid_confirmed.describe()
no_cases = data_corona[data_corona['Confirmed']==0]

print(no_cases.count())

          
no_cases = data_corona[data_corona['Recovered']!=0]

no_cases

          
#expect china display remaining countries



no_china=data_corona=data_corona[data_corona['Country']!='China']



no_china.head(20)

no_china
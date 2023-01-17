import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px

import plotly.offline as py

py.init_notebook_mode(connected=True)

import datetime



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
co_india = pd.read_csv("../input/covid19-in-india/covid_19_india.csv")

age = pd.read_csv("../input/covid19-in-india/AgeGroupDetails.csv")

co_india.rename(columns={'State/UnionTerritory':'state'},inplace=True)

co_india
# data_ts= co_india.query("state == 'Telengana'")

fig = px.bar(co_india, x='state', y='Confirmed',hover_data=['Cured', 'Deaths'],animation_frame="Date", color='Confirmed')

fig.update_layout(

    title="Date wise change in number of confirmed cases of different states",

    font=dict(

        family="Courier New, monospace",

    )

)

fig.show()
co_india['Date'] = pd.to_datetime(co_india['Date'])

co_india['Total Cases'] = co_india['Cured'] + co_india['Deaths'] + co_india['Confirmed']

co_india['active']= co_india['Confirmed'] - co_india['Deaths'] - co_india['Cured']
# state_details['Recovery Rate'] = round(state_details['Cured'] / state_details['Confirmed'],3)

# state_details['Death Rate'] = round(state_details['Deaths'] /state_details['Confirmed'], 3)

state_details = pd.pivot_table(co_india, values=['Total Cases','Confirmed','Deaths','Cured'], index='state', aggfunc='max')

state_details['Recovery Rate'] = round(state_details['Cured'] / state_details['Confirmed'],3)

state_details['Death Rate'] = round(state_details['Deaths'] /state_details['Confirmed'], 3)

state_details = state_details.sort_values(by='Total Cases', ascending= False)

state_details.style.background_gradient(cmap='Greens')

labels = list(age['AgeGroup'])

sizes = list(age['TotalCases'])

# print(labels)

# print(sizes)

    

plt.figure(figsize= (15,10))

plt.title('% Age group affected')

plt.pie(sizes, labels=labels)

plt.show()
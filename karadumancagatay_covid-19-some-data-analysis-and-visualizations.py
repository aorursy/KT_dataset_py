# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")
data.head()
data.columns
data = data.rename(columns={'Country/Region': 'Country',"Province/State":"Province"})
data.columns
data.info()
data.describe()
deaths= data[(data["Deaths"]>1000)]
plt.figure(figsize=(15,15))

sns.barplot(x=deaths['Country'], y=deaths['Deaths'])

plt.xticks(rotation= 45)

plt.xlabel('Country')

plt.ylabel('Deaths')

plt.title('COVID-19 | Deaths over 1000 Deaths According to Country')

plt.show()
deaths_most= data[(data["Deaths"] > data["Recovered"]) & (data["Deaths"]>1000)]
plt.figure(figsize=(15,15))

sns.barplot(x=deaths_most['Country'], y=deaths_most['Deaths'])

plt.xticks(rotation= 0)

plt.xlabel('Country')

plt.ylabel('Deaths')

plt.title('COVID-19 | Deaths over Recovered and 1000 Deaths According to Country')

plt.show()
recovered_most = data[(data["Recovered"]>10000)]

recovered_most
plt.figure(figsize=(15,15))

sns.barplot(x=recovered_most['Country'], y=recovered_most['Recovered'])

plt.xticks(rotation= 90)

plt.xlabel('Country')

plt.ylabel('Recovered')

plt.title('COVID-19 | Recovered over 10000 According to Country')

plt.show()
trace1 = go.Bar(

                x = deaths.Country,

                y = deaths.Deaths,

                name = "Deaths",

                marker = dict(color = 'rgba(255, 174, 255, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=0.02)),

                text = deaths.Country) 

trace2 = go.Bar(

                x = deaths.Country,

                y = deaths.Recovered,

                name = "Recovered",

                marker = dict(color = 'rgba(255, 255, 128, 0.5)',

                              line=dict(color='rgb(0,0,0)',width=0.02)),

                text = deaths.Country)

data = [trace1, trace2]

layout = go.Layout(barmode = "group")

fig = go.Figure(data = data, layout = layout)

iplot(fig)
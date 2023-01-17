# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os
AgeGroupDetails=pd.read_csv("/kaggle/input/covid19-in-india/AgeGroupDetails.csv")

AgeGroupDetails.head()
sns.set(rc={'figure.figsize':(11,8)})

x=AgeGroupDetails.AgeGroup

y=AgeGroupDetails.TotalCases

y_pos=np.arange(len(x))

plt.xticks(y_pos,x)

plt.xticks(rotation=90)

plt.xlabel('Age Groups')

ax=sns.kdeplot(y_pos,y,cmap='Blues',shade=True,cbar=True)
covid_19_india=pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')

covid_19_india.head()
sns.pairplot(covid_19_india, palette="Set2")
from IPython.core.display import HTML

import folium

import datetime

from datetime import datetime

import requests

from bs4 import BeautifulSoup

import lxml.html as lh

import pandas as pd

import re

import time

import psutil

import json



import numpy as np

from PIL import Image

import os

from os import path

import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp



import plotly.graph_objects as go

from pandas.plotting import register_matplotlib_converters

import plotly.express as px

from IPython.display import display, Markdown, Latex

import matplotlib as plot

from matplotlib.pyplot import figure

import seaborn as sns



register_matplotlib_converters()

from IPython.display import Markdown





dataset = pd.DataFrame()
indiaLiveJson = 'https://api.covid19india.org/data.json'

r = requests.get(indiaLiveJson)

indiaData = r.json()
testingHistory = pd.DataFrame()

testingNO = []

testedPos = []

timeStamp = []

for index in range(len(indiaData['tested'])):

    try:

        testingNO.append(int(re.sub(',','',indiaData['tested'][index]['totalindividualstested'])))

        testedPos.append(int(re.sub(',','',indiaData['tested'][index]['totalpositivecases'])))

    except:

        testingNO.append(testingNO[len(testingNO)-1])

        testedPos.append(testedPos[len(testedPos)-1])

        

    timeStamp.append(indiaData['tested'][index]['updatetimestamp'][:-9])

    

testingHistory['testing_no'] = testingNO[:-1]

testingHistory['testing_pos'] = testedPos

testingHistory['time_stamp'] = timeStamp



testingHistory.drop_duplicates(subset ="time_stamp", 

                     keep = False, inplace = True) 





fig = go.Figure()



fig = fig.add_trace(go.Scatter(y=testingHistory['testing_no'], x=testingHistory['time_stamp'],

                    mode='lines+markers',

                    name='Testing Pattern'))



fig = fig.add_trace(go.Scatter(y=testingHistory['testing_pos'], x=testingHistory['time_stamp'],

                    mode='lines+markers',

                    name='Tested Positive'))



fig = fig.update_layout(

    title="India COVID-19 Testing History",

    xaxis_title="Testing",

    yaxis_title="Date",

    

)





fig.show()
indiaConfirmed = []

indiaRecovered = []

indiaDeseased = []

timeStamp = []

for index in range(len(indiaData['cases_time_series'])):

    indiaConfirmed.append(int(re.sub(',','',indiaData['cases_time_series'][index]['totalconfirmed'])))

    indiaRecovered.append(int(re.sub(',','',indiaData['cases_time_series'][index]['totalrecovered'])))

    indiaDeseased.append(int(re.sub(',','',indiaData['cases_time_series'][index]['totaldeceased'])))

    

    timeStamp.append(indiaData['cases_time_series'][index]['date'])

    



fig = go.Figure()

#fig = px.scatter(testingHistory,x="time_stamp", y="testing_no")



fig = fig.add_trace(go.Scatter(x=timeStamp, y=indiaConfirmed,

                    mode='lines+markers',

                    name='Confirmed Cases'))

fig = fig.add_trace(go.Scatter(x=timeStamp, y=indiaRecovered,

                    mode='lines+markers',

                    name='Recoverd Patients'))

fig = fig.add_trace(go.Scatter(x=timeStamp, y=indiaDeseased,

                    mode='lines+markers',

                    name='Deseased Patients'))



fig = fig.update_layout(

    title="India COVID-19 ",

    xaxis_title="Date",

    yaxis_title="Testing",

    

)





fig.show()
plt.figure(figsize=(5,5))

cured=covid_19_india[covid_19_india['Cured']==True]

deaths=covid_19_india[covid_19_india['Deaths']==True]

slices_hours = [cured['Time'].count(),deaths['Time'].count()]

activities = ['Cured', 'Deaths']

colors = ['red', 'green']

explode=(0,0.1)

plt.pie(slices_hours, labels=activities,explode=explode, colors=colors, startangle=90, autopct='%1.1f%%',shadow=True)

plt.show()
covid_19_india['active']=covid_19_india['Confirmed']-(covid_19_india['Cured']+covid_19_india['Deaths'])

f,axes = plt.subplots(2, 2, figsize=(15,10))

sns.distplot( covid_19_india["Cured"] , color="blue", ax=axes[0, 0])

sns.distplot( covid_19_india["Deaths"] , color="violet", ax=axes[0, 1])

sns.distplot( covid_19_india["Confirmed"] , color="olive", ax=axes[1, 0])

sns.distplot( covid_19_india["active"] , color="orange", ax=axes[1, 1])

f.subplots_adjust(hspace=.3,wspace=0.03) 
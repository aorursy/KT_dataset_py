# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd 
import os
import time
from datetime import date, datetime
from dateutil import parser
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
## Load in dataset, and see information ##
dataset = pd.read_csv("../input/daily-inmates-in-custody.csv")
print(dataset.info())
print(dataset.head(15))
for c in dataset.columns:
    print("---",c,"---")
    x = dataset[c].value_counts() 
    print(x)
ages=[]
for i in dataset['AGE']:
    ages.append(i)
ages = pd.DataFrame(ages)
ages.describe()

sns.set_style("whitegrid")
plotOne = sns.FacetGrid(ages,aspect=2.5)
plotOne.map(sns.kdeplot,0,shade=True)
plotOne.set(xlim=(10, 75))
plotOne.set_axis_labels('Age', 'Proportion')
plotOne.fig.suptitle('Age of Inmates in NY')
plt.show()

sns.set_style("whitegrid")
plotOne = sns.FacetGrid(ages,aspect=2.5)
plotOne.map(sns.distplot,0,kde=False)
plotOne.set(xlim=(10, 75))
plotOne.set_axis_labels('Age', 'Count')
plotOne.fig.suptitle('Age of Inmates in NY')
plt.show()
daysInJail=[]
today = datetime.now().date()
for i in dataset['ADMITTED_DT']:
    dateOfArrival_0 = parser.parse(i)
    jailday = dateOfArrival_0.date()
    diff = jailday - today
    a= diff.days
    b= -a
    daysInJail.append(b)
daysInJail = pd.DataFrame(daysInJail)

sns.set_style("whitegrid")
plotOne = sns.FacetGrid(daysInJail,aspect=2.5)
plotOne.map(sns.kdeplot,0,shade=True)
plotOne.set(xlim=(0, 2000))
plotOne.set_axis_labels('Number of Days', 'Proportion')
plotOne.fig.suptitle('How Long Inmates Have Been Detained in NY')
plt.show()

sns.set_style("whitegrid")
plotOne = sns.FacetGrid(daysInJail,aspect=2.5)
plotOne.map(sns.distplot,0,kde=False)
plotOne.set(xlim=(0, 2000))
plotOne.set_axis_labels('Number of Days', 'Count')
plotOne.fig.suptitle('How Long Inmates Have Been Detained in NY')
plt.show()
GENDER = dataset['GENDER'].value_counts()
gender = pd.DataFrame(GENDER)
gender['Gender'] = gender.index
gender = gender[['Gender', 'GENDER']]

trace1 = go.Bar(
                x = gender.Gender,
                y = gender.GENDER,
                name = "citations",
                marker = dict(color = 'rgba(0, 0, 255, 0.8)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = gender.Gender)
data = [trace1]
layout = go.Layout(barmode = "group",title='Gender of Inmates in NY')
fig = go.Figure(data = data, layout = layout)
iplot(fig)
RACE = dataset['RACE'].value_counts()
print(RACE)
race = pd.DataFrame(RACE)
race['Race'] = race.index
race = race[['Race', 'RACE']]
print(race)

trace1 = go.Bar(
                x = race.Race,
                y = race.RACE,
                name = "citations",
                marker = dict(color = 'rgba(0, 0, 255, 0.8)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = race.Race)
data = [trace1]
layout = go.Layout(barmode = "group",title='Race of Inmates in NY')
fig = go.Figure(data = data, layout = layout)
iplot(fig)
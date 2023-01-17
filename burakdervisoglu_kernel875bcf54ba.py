# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# plotly

import plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import seaborn as sns



# word cloud library

from wordcloud import WordCloud



# matplotlib

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



# Any results you write to the current directory are saved as output.

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
crime = pd.read_csv('../input/crime.csv',encoding = 'latin-1')
crime.head()
crime.info()
crime.shape
crime.describe()
crime.shape
crime.drop(columns = {'INCIDENT_NUMBER','INCIDENT_NUMBER','Long','Location','Lat'},axis = 1)
plt.subplots(figsize = (15,9))

sns.barplot(crime.OFFENSE_DESCRIPTION.value_counts()[:20].index,crime.OFFENSE_DESCRIPTION.value_counts()[:20])

plt.xticks(rotation = 90)

plt.show()
crime2018 = crime[crime.YEAR == 2018]

crime2017 = crime[crime.YEAR == 2017]

crime2016 = crime[crime.YEAR == 2016]

crime2015 = crime[crime.YEAR == 2015]

trace1 = go.Bar(

        x = crime2018.YEAR.iloc[:20],

        y = crime2018.REPORTING_AREA.value_counts()[:20],

        marker = dict(color = 'red'),

        name = '2018')

trace2 = go.Bar(

        x = crime2017.YEAR.iloc[:20],

        y = crime2017.REPORTING_AREA.value_counts()[:20],

        marker = dict(color = 'blue'),

        name = '2017')

trace3 = go.Bar(

        x = crime2016.YEAR[:20],

        y = crime2016.REPORTING_AREA.value_counts()[:20],

        marker = dict(color = 'yellow'),

        name = '2016')

trace4 = go.Bar(

            x = crime2015.YEAR,

            y = crime2015.REPORTING_AREA.value_counts()[:20],

            marker = dict(color = 'brown'),

            name = '2015')



layout = go.Layout(barmode = 'group')

data = [trace1,trace2,trace3,trace4]

fig = go.Figure(data,layout)

iplot(fig)
plt.subplots(figsize = (12,12))

crime.OFFENSE_CODE_GROUP.value_counts()[:20].sort_values().plot(kind = 'barh')

plt.show()
trace1 = go.Bar(

        x = crime2018.OFFENSE_CODE_GROUP.value_counts().index[:20],

        y = crime2018.OFFENSE_CODE_GROUP.value_counts().values[:20],

        

        marker = dict(color = 'red'),

        name = '2018')

trace2 = go.Bar(

        x = crime2017.OFFENSE_CODE_GROUP.value_counts().index[:20],

        y = crime2017.OFFENSE_CODE_GROUP.value_counts().values[:20],

        

        marker = dict(color = 'blue'),

        name = '2017')

trace3 = go.Bar(

        x = crime2016.OFFENSE_CODE_GROUP.value_counts().index[:20],

        y = crime2016.OFFENSE_CODE_GROUP.value_counts().values[:20],

        

        marker = dict(color = 'yellow'),

        name = '2016')

trace4 = go.Bar(

         x = crime2015.OFFENSE_CODE_GROUP.value_counts().index[:20],

         y = crime2015.OFFENSE_CODE_GROUP.value_counts().values[:20],

            

         marker = dict(color = 'brown'),

         name = '2015')



layout = go.Layout(barmode = "relative",title =" ")

data = [trace1,trace2,trace3,trace4]

fig = go.Figure(data,layout)

iplot(fig)
plt.subplots(figsize = (15,6))

crime.DAY_OF_WEEK.value_counts().sort_values().plot(kind = 'bar')

plt.show()
plt.subplots(figsize = (15,6))

crime.HOUR.value_counts().sort_values().plot(kind = 'bar')

plt.show()
plt.subplots(figsize = (15,6))

crime.MONTH.value_counts().sort_values().plot(kind = 'bar')

plt.show()
trace1 = go.Bar(

        x = crime2015.MONTH.value_counts().index[:20],

        y = crime2015.MONTH.value_counts().values[:20],

        

        marker = dict(color = 'red'),

        name = '2018')

trace2 = go.Bar(

        x = crime2016.MONTH.value_counts().index[:20],

        y = crime2016.MONTH.value_counts().values[:20],

        

        marker = dict(color = 'blue'),

        name = '2017')

trace3 = go.Bar(

        x = crime2017.MONTH.value_counts().index[:20],

        y = crime2017.MONTH.value_counts().values[:20],

        

        marker = dict(color = 'yellow'),

        name = '2016')

trace4 = go.Bar(

         x = crime2018.MONTH.value_counts().index[:20],

         y = crime2018.MONTH.value_counts().values[:20],

            

         marker = dict(color = 'brown'),

         name = '2015')



layout = go.Layout(barmode = "relative",title =" ")

data = [trace1,trace2,trace3,trace4]

fig = go.Figure(data,layout)

iplot(fig)
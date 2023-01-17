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
train=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv")

test=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/test.csv")

submission=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/submission.csv")

print(train.shape)

train.head()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

italy=train[train["Country/Region"]=="Italy"]

plt.figure(figsize=(20,15))

ax = sns.lineplot(x="Date", y="Fatalities", data=italy)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

turkey=train[train["Country/Region"]=="Turkey"]

plt.figure(figsize=(15,30))

ax = sns.lineplot(x="Date", y="Fatalities", data=turkey)
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

japan=train[train["Country/Region"]=="Japan"]

plt.figure(figsize=(15,30))

ax = sns.lineplot(x="Date", y="Fatalities", data=japan)
import plotly.express as px

fig = px.choropleth(train, 

                    locations="Country/Region", 

                    locationmode = "country names", 

                    color="Fatalities", 

                    hover_name='Country/Region', 

                    animation_frame="Date"

                   )



fig.update_layout(

    title_text = 'Corona Spreading',

    title_x = 0.5,

    geo=dict(

        showframe = False,

        showcoastlines = False,

    ))

fig
set(train['Country/Region'])
import plotly.express as px

fig = px.choropleth(train, 

                    locations="Country/Region", 

                    locationmode = "country names", 

                    color='ConfirmedCases', 

                    hover_name='Country/Region', 

                    animation_frame="Date"

                   )



fig.update_layout(

    title_text = 'Corona Spreading',

    title_x = 0.5,

    geo=dict(

        showframe = False,

        showcoastlines = False,

    ))

fig
fig = px.choropleth(train, 

                    locations="Country/Region", 

                    locationmode = "country names", 

                    color='ConfirmedCases',

                    range_color=(0,6000),

                    hover_name='Country/Region', 

                    animation_frame="Date"

                   )



fig.update_layout(

    title_text = 'Corona Spreading',

    title_x = 0.5,

    geo=dict(

        showframe = False,

        showcoastlines = False,

    ))

fig
fig = px.choropleth(train[train["Date"]=="2020-03-02"],

                    locations="Country/Region", 

                    locationmode = "country names",

                    range_color=(0,50),

                    color="ConfirmedCases",

                    hover_name="Country/Region",

                    color_continuous_scale=px.colors.sequential.Plasma)

fig
fig = px.choropleth(train[train["Date"]=="2020-03-10"],

                    locations="Country/Region", 

                    locationmode = "country names",

                    range_color=(0,50),

                    color="ConfirmedCases",

                    hover_name="Country/Region",

                    color_continuous_scale=px.colors.sequential.Plasma)

fig
fig = px.choropleth(train, 

                    locations="Country/Region", 

                    locationmode = "country names", 

                    color='ConfirmedCases',

                    range_color=(0,2000),

                    hover_name='Country/Region', 

                    animation_frame="Date"

                   )



fig.update_layout(

    title_text = 'Corona Spreading',

    title_x = 0.5,

    geo=dict(

        showframe = False,

        showcoastlines = False,

    ))

fig
fig = px.choropleth(train, 

                    locations="Country/Region", 

                    locationmode = "country names", 

                    color='ConfirmedCases',

                    range_color=(0,40),

                    hover_name='Country/Region', 

                    animation_frame="Date"

                   )



fig.update_layout(

    title_text = 'Corona Spreading',

    title_x = 0.5,

    geo=dict(

        showframe = False,

        showcoastlines = False,

    ))

fig
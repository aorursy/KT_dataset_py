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
import pandas as pd

import numpy as np

from sklearn import linear_model

from sklearn import metrics

%matplotlib inline 

from sklearn.utils import shuffle

import matplotlib.pyplot as plt

from numpy import array

from numpy import argmax

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_selection import chi2

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

import seaborn as sns

import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_dark"
df=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv', na_filter=False)

df=df.drop(['Id'],axis=1)

df.head(10)
#Bangladesh

bd=df.loc[df['Country/Region'] == 'Bangladesh']

#China

cn=df.loc[df['Country/Region'] == 'China']

#Italy

it=df.loc[df['Country/Region'] == 'Italy']

#US

us=df.loc[df['Country/Region'] == 'US']

bd.tail(10)
dt=df.loc[df['Date'] == '2020-03-24']

dt.head(50)

#data=dt.groupby(["Country/Region"])['ConfirmedCases'].count()

#data.head(50)

#df.groupby('Country/Region').plot(x='Date', y='ConfirmedCases')
dt_confirmed = dt[dt["ConfirmedCases"]>=1]

dt_confirmed=dt_confirmed[["Country/Region","Date","ConfirmedCases"]]

dt_confirmed.head(50)
dn=dt_confirmed.groupby(['Country/Region'])['ConfirmedCases'].sum() 

dn.head(50)
dn.sort_values(ascending=True).plot(x='Country/Region',y='ConfirmedCases',fontsize=18,figsize=(20,100),kind='barh');
group = df.groupby('Date')['ConfirmedCases'].sum().reset_index()



fig = px.line(group, x="Date", y="ConfirmedCases", 

              title="Worldwide Confirmed Cases Over Time")



fig.show()
group = df.groupby('Date')['Fatalities'].sum().reset_index()



fig = px.line(group, x="Date", y="Fatalities", 

              title="Worldwide Fatalities Over Time")



fig.show()
bd.sort_values(by=['Date'],ascending=True).plot(x='Date',y='ConfirmedCases',fontsize=15,figsize=(12,18), kind='barh');
group = bd.groupby('Date')['Date','ConfirmedCases'].sum().reset_index()



fig = px.line(group, x="Date", y="ConfirmedCases", 

              title="Confirmed Cases for Bangladesh Over Time")



fig.show()
group = cn.groupby('Date')['Date','ConfirmedCases'].sum().reset_index()



fig = px.line(group, x="Date", y="ConfirmedCases", 

              title="Confirmed Cases for China Over Time")



fig.show()
group = it.groupby('Date')['Date','ConfirmedCases'].sum().reset_index()



fig = px.line(group, x="Date", y="ConfirmedCases", 

              title="Confirmed Cases for Italy Over Time")



fig.show()
group = us.groupby('Date')['Date','ConfirmedCases'].sum().reset_index()



fig = px.line(group, x="Date", y="ConfirmedCases", 

              title="Confirmed Cases for US Over Time")



fig.show()
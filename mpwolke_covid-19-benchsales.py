#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://scontent-yyz1-1.cdninstagram.com/v/t51.2885-15/sh0.08/e35/s640x640/94319744_810086362847676_153750449687750353_n.jpg?_nc_ht=scontent-yyz1-1.cdninstagram.com&_nc_cat=100&_nc_ohc=hTjdHBOqZVIAX8yut63&oh=06295dde467edc06cf48522640b7ea54&oe=5ECBEC1A',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

import seaborn



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcT3oQ2XuXEwOIPSY2yDoHpqnPuGWwtOLEAAGhC62YpaNSUEgNJX&usqp=CAU',width=400,height=400)
df = pd.read_excel('/kaggle/input/covid-19-monthly-sales-report-of-companies-mar-20/benchsales19.xls')

df.head()
df = df.rename(columns={'Unnamed: 1':'unnamed', 'Unnamed: 2': 'unnamed1', 'Estimates of Monthly Retail and Food Services Sales by Kind of Business: 2019' : 'sales', 'Unnamed: 3' : 'unnamed2', 'Unnamed: 4': 'unnamed3', 'Unnamed: 5': 'unnamed4', 'Unnamed: 6': 'unnamed5'})
fig = px.bar(df, 

             x='sales', y='unnamed', color_discrete_sequence=['#27F1E7'],

             title='Covid-19 & Benchsales', text='sales')

fig.show()
fig = px.bar(df, 

             x='sales', y='unnamed5', color_discrete_sequence=['crimson'],

             title='Covid-19 & Benchsales', text='sales')

fig.show()
fig = px.density_contour(df, x="sales", y="unnamed", color_discrete_sequence=['purple'])

fig.show()
fig = px.density_contour(df, x="sales", y="unnamed3", color_discrete_sequence=['crimson'])

fig.show()
fig = px.scatter(df.dropna(), x='sales',y='unnamed', trendline="unnamed4", color_discrete_sequence=['purple'])

fig.show()
seaborn.set(rc={'axes.facecolor':'#27F1E7', 'figure.facecolor':'#27F1E7'})

sns.countplot(df["unnamed"])

plt.xticks(rotation=90)

plt.show()
fig = px.line(df, x="sales", y="unnamed", color_discrete_sequence=['green'], 

              title="Covid-19 & Benchsales")

fig.show()
fig = px.line(df, x="unnamed5", y="unnamed", color_discrete_sequence=['black'], 

              title="Covid-19 & Benchsales")

fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRwSCDSXCJOLbjqwWEo0C0Uoxl6wvXi2p12PVbRZvg9f4ajuSWd&usqp=CAU',width=400,height=400)
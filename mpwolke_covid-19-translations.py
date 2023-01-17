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
df = pd.read_csv('../input/covid19-translations/COVID-19 Translations - Quarrantine.tsv', sep='\t')

df.head()
fig = px.bar(df, 

             x='English', y='Telgugu', color_discrete_sequence=['#27F1E7'],

             title='Translation Telugu', text='English')

fig.show()
fig = px.bar(df, 

             x='Tamil', y='Khmer', color_discrete_sequence=['crimson'],

             title='Translation Khmer', text='Tamil')

fig.show()
px.histogram(df, x='Malayalam', color='Bengali')
fig = px.line(df, x="English", y="Bengali", 

              title="Covid-19 Translations")

fig.show()
seaborn.set(rc={'axes.facecolor':'#27F1E7', 'figure.facecolor':'#27F1E7'})

sns.countplot(df["English"])

plt.xticks(rotation=90)

plt.show()
fig = px.scatter(df.dropna(), x='English',y='Marathi', trendline="Panjabi", color_discrete_sequence=['purple'])

fig.show()
fig = px.density_contour(df, x="English", y="Burmese", color_discrete_sequence=['purple'])

fig.show()
px.histogram(df, x='English', color='Panjabi')
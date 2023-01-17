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



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/cord19-study-design/attribute.csv")

df.head()
px.histogram(df, x='text', color='label')
fig = px.bar(df, 

             x='label', y='text', color_discrete_sequence=['#D63230'],

             title='Covid-19 Study Design', text='label')

fig.show()
# Set the width and height of the figure

plt.figure(figsize=(10,6))



# Add title

plt.title("Covid-19 Study Design")



# Bar chart showing average arrival delay for Spirit Airlines flights by month

sns.barplot(x=df.index, y=df['label'])



# Add label for vertical axis

plt.ylabel("Covid-19 Study Design")
fig = px.line(df, x="text", y="label", 

              title="Covid-19 Study Design")

fig.show()
fig = px.line(df, 

             x='text', y='label', color_discrete_sequence=['#D63230'],

             title='Covid-19 Study Design', text='label')

fig.show()
df["label"].plot.hist()

plt.show()
sns.countplot(df["label"])

plt.xticks(rotation=90)

plt.show()
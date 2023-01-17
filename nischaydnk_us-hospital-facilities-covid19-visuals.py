# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.express as px

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/hospital-bed-capacity-and-covid19/HRR Scorecard_ 20 _ 40 _ 60 - 20 Population.csv")

data = data.drop(0)

data.head(10)
data.columns
data = data.replace(',','', regex=True)

data = data.replace('%','', regex=True)



data
data['Available Hospital Beds'].unique()
data1 = data.drop(['HRR'],axis=1)

data1 = data1.apply(pd.to_numeric)

data1
data2 = data['HRR']

data3 = pd.concat([data2, data1], axis=1, join='inner')

data = data3
fig = px.treemap(data, path=['HRR'], values='Adult Population',

                  color='Adult Population', hover_data=['HRR'],

                  color_continuous_scale='dense', title='Current Adult(18+) Population in different Locations of US ')

fig.show()
fig = px.treemap(data, path=['HRR'], values='Total Hospital Beds',

                  color='Adult Population', hover_data=['HRR'],

                  color_continuous_scale='dense', title='Total Available Beds in different Locations.')

fig.show()
fig = px.treemap(data, path=['HRR'], values='Total ICU Beds',

                  color='Adult Population', hover_data=['HRR'],

                  color_continuous_scale='dense', title='Total Available Beds in different Locations.')

fig.show()
fig = px.treemap(data, path=['HRR'], values='Percentage of Potentially Available Beds Needed, Six Months',

                  color='Adult Population', hover_data=['HRR'],

                  color_continuous_scale='dense', title='Percentage of Potentially Available Beds Needed in next 6 Months')

fig.show()
fig = px.treemap(data, path=['HRR'], values='Percentage of Potentially Available Beds Needed, Twelve Months',

                  color='Adult Population', hover_data=['HRR'],

                  color_continuous_scale='dense', title='Percentage of Potentially Available Beds Needed in next 12 Months')

fig.show()
fig = px.treemap(data, path=['HRR'], values='Percentage of Potentially Available Beds Needed, Eighteen Months',

                  color='Adult Population', hover_data=['HRR'],

                  color_continuous_scale='dense', title='Percentage of Potentially Available Beds Needed in next 18 Months')

fig.show()
fig1data = data.sort_values(by = ['Adult Population'],ascending = False).head(15)
fig1 = px.pie(fig1data, values='Adult Population', names='HRR')

fig1.update_traces(rotation=90, pull=0.1, textinfo="value")



fig1.update_layout(uniformtext_minsize=12, uniformtext_mode='show')

fig1.show()
fig2data = data.sort_values(by = ['Available Hospital Beds'],ascending='False').head(15)

fig2 = px.pie(fig1data, values='Available Hospital Beds', names='HRR')

fig2.update_traces(rotation=90, pull=0.1, textinfo="value")

fig2.update_layout(uniformtext_minsize=12, uniformtext_mode='show')

fig2.show()
fig3data = data.sort_values(by = ['Available ICU Beds'],ascending='False').head(20)

fig3 = px.pie(fig1data, values='Available ICU Beds', names='HRR')

fig3.update_layout(uniformtext_minsize=12, uniformtext_mode='show')

fig3.update_traces(rotation=90, pull=0.1, textinfo="value")



fig3.show()
fig3data = data.sort_values(by = ['Projected Infected Individuals'],ascending='False').head(20)

fig3 = px.pie(fig1data, values='Projected Infected Individuals', names='HRR')

fig3.update_layout(uniformtext_minsize=12, uniformtext_mode='show')

fig3.update_traces(rotation=90, pull=0.1, textinfo="value")



fig3.show()
data['Ratio of Infected Individuals per Available beds'] = data['Projected Hospitalized Individuals'] / data['Available Hospital Beds']
fig = px.treemap(data, path=['HRR'], values='Ratio of Infected Individuals per Available beds',

                  color='Adult Population', hover_data=['HRR'],

                  color_continuous_scale='dense', title='Higher the ratio, More chances of suuffering.')

fig.show()
fig5data = data.sort_values(by = ['Ratio of Infected Individuals per Available beds'],ascending=False).head(15)

fig5 = px.pie(fig5data, values='Ratio of Infected Individuals per Available beds', names='HRR')

fig5.update_traces(textposition='inside')

fig5.update_layout(uniformtext_minsize=12, uniformtext_mode='show')

fig5.update_traces(rotation=90, pull=0.1, textinfo="value")



fig5.show()
import seaborn as sns

fig_dims = (20, 8)

fig, ax = plt.subplots(figsize=fig_dims)

sns.barplot(x = fig5data['Ratio of Infected Individuals per Available beds'],ax=ax, y=fig5data['HRR'])
fig5data = data.sort_values(by = ['Ratio of Infected Individuals per Available beds'],ascending=True).head(15)

fig5 = px.pie(fig5data, values='Ratio of Infected Individuals per Available beds', names='HRR')

fig5.update_traces(textposition='inside')

fig5.update_layout(uniformtext_minsize=12, uniformtext_mode='show')

fig5.update_traces(rotation=90, pull=0.1, textinfo="value")



fig5.show()
import seaborn as sns

fig_dims = (20, 8)

fig, ax = plt.subplots(figsize=fig_dims)

sns.barplot(x = fig5data['Ratio of Infected Individuals per Available beds'],ax=ax, y=fig5data['HRR'])
data['Ratio of Individuals that need ICU care per Available ICU beds'] = data['Projected Individuals Needing ICU Care'] / data['Available ICU Beds']

data
fig = px.treemap(data, path=['HRR'], values='Ratio of Individuals that need ICU care per Available ICU beds',

                  color='Adult Population', hover_data=['HRR'],

                  color_continuous_scale='dense', title='Higher the ratio, More chances of suuffering.')

fig.show()
fig5data = data.sort_values(by = ['Ratio of Individuals that need ICU care per Available ICU beds'],ascending=False).head(15)

fig5 = px.pie(fig5data, values='Ratio of Individuals that need ICU care per Available ICU beds', names='HRR')

fig5.update_traces(textposition='inside')

fig5.update_layout(uniformtext_minsize=12, uniformtext_mode='show')

fig5.update_traces(rotation=90, pull=0.1, textinfo="value")



fig5.show()
import seaborn as sns

fig_dims = (20, 8)

fig, ax = plt.subplots(figsize=fig_dims)

sns.barplot(x = fig5data['Ratio of Individuals that need ICU care per Available ICU beds'],ax=ax, y=fig5data['HRR'])
fig5data = data.sort_values(by = ['Ratio of Individuals that need ICU care per Available ICU beds'],ascending=True).head(15)

fig5 = px.pie(fig5data, values='Ratio of Individuals that need ICU care per Available ICU beds', names='HRR')

fig5.update_traces(textposition='inside')

fig5.update_layout(uniformtext_minsize=12, uniformtext_mode='show')

fig5.update_traces(rotation=90, pull=0.1, textinfo="value")



fig5.show()
import seaborn as sns

fig_dims = (20, 8)

fig, ax = plt.subplots(figsize=fig_dims)

sns.barplot(x = fig5data['Ratio of Individuals that need ICU care per Available ICU beds'],ax=ax, y=fig5data['HRR'])
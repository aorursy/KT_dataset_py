# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import plotly.graph_objs as go



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Import datasets

timeline = pd.read_csv("/kaggle/input/international-mathematical-olynpiad-results/timeline.csv")

results = pd.read_csv("/kaggle/input/international-mathematical-olynpiad-results/results.csv")

countries = pd.read_csv("/kaggle/input/international-mathematical-olynpiad-results/countries.csv")
timeline.head()
results.head()
countries.head()
timeline['Year'].describe()
#We fill in the missing values

timeline.fillna(0, inplace=True)



#Add the Unknown Gender column

timeline['Unknown_Gender'] = timeline.All - (timeline.F + timeline.M)
timeline = timeline[timeline['Year'] < 2020]

imo_participation_2019 = timeline[timeline.Year == 2019]

imo_participation_2019 = imo_participation_2019[['M', 'F']] 

imo_participation_2019 = pd.melt(imo_participation_2019)
fig = px.pie(imo_participation_2019, values='value', names='variable', title='Female Participation IMO 2019')

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=timeline['Year'], y=timeline['M'],

                    mode='lines',

                    name='Males'))



fig.add_trace(go.Scatter(x=timeline['Year'], y=timeline['F'],

                    mode='lines',

                    name='Females'))



fig.add_trace(go.Scatter(x=timeline['Year'], y=timeline['Unknown_Gender'],

                    mode='lines',

                    name='Unknown Gender'))



fig.update_layout(title='Participation to IMO by Gender')



fig.show()
results.rename(columns={'Year': 'Country'}, inplace=True)

results.head()
countries_dict = dict(zip(countries.Code, countries.Country))

results['Country_Name'] = results['Country'].map(countries_dict)
fig = px.choropleth(results, locations="Country",

                    color="19", # lifeExp is a column of gapminder

                    hover_name="Country_Name", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Plasma)



fig.update_layout(title='Results IMO 2019')

fig.show()
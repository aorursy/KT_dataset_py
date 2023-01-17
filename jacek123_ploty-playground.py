# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

import pycountry

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')

df.head()
df.describe()
sample_data=df.sample(n=1000,replace="False")
sns.lineplot(data=sample_data, x='year', y='suicides/100k pop')

plt.title('Suicides over taime')

plt.show()

bins = np.linspace(sample_data['gdp_per_capita ($)'].min(), sample_data['gdp_per_capita ($)'].max(), 12)

sample_data['binned_gbp'] = pd.cut(df['gdp_per_capita ($)'], bins)
bins = np.linspace(sample_data['population'].min(), sample_data['population'].max(), 10)

sample_data['binned_population'] = pd.cut(df['population'], bins)
fig = px.box(sample_data, x="country", y="suicides/100k pop", color="sex")

fig.show()
fig = px.scatter(sample_data, x="age", y="suicides/100k pop", marginal_y="histogram")

fig.show()
test = df.groupby(['year','age','sex'], as_index=False)["suicides/100k pop"].mean()

fig = px.line(test, x="year", y="suicides/100k pop", color="sex", facet_col="age")

fig.show()
df = df[df['year'] < 2016]

df = df.replace('5-14 years','05-14 years')
year_age = df.sort_values(['year', 'age'], ascending=[True, True])

px.strip(year_age, x="age", y="suicides/100k pop", color="sex", animation_frame="year")
fig = px.histogram(df, x="year", y="suicides/100k pop", color="generation", marginal="box", histnorm="percent")

fig.show()
input_countries = df['country']



countries = {}

for country in pycountry.countries:

    countries[country.name] = country.alpha_3



codes = [countries.get(country, 'Unknown code') for country in input_countries]



df['country_codes'] = codes

year_asc = df.sort_values('year', ascending=True)

fig = px.choropleth(

    year_asc,

    locations="country_codes",

    color="suicides/100k pop",

    hover_name="country",

    animation_frame="year",

    range_color=[0,100],

    color_continuous_scale=px.colors.diverging.RdYlGn[::-1])

fig.show()
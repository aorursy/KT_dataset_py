# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input/'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



pd.set_option('display.max_columns', 1000)

base = "/kaggle/input/corruption-perceptions-index-for-10-years"



# Any results you write to the current directory are saved as output.
df = pd.read_csv(f"{base}/merged_cpi_data.csv")

df
def draw_line_plot(x, y, hue, data, title):

    fig = plt.figure(figsize=(15, 8))

    sns.lineplot(x=x, y=y, hue=hue, data=data).set_title(title)
orgs = ["G20", "BRICS", "EU", "Arab states"]

for org in orgs:

    brics_data = df[df[org] == 'y'][['Year', 'Country', 'CPI Score']]

    draw_line_plot(x='Year', y='CPI Score', hue='Country', data=brics_data, title=f"Corruption for {org}")
region_map = {'AME': 'Americas', 'AP': 'Asia Pacific', 'ECA': 'Europe & Central Asia', 

              'MENA': 'Middle East & North Africa', 'WE/EU': 'Western Europe/European Union',

              'SSA': 'Sub-saharan Africa'}

avg_scores_by_year_region = df.groupby(['Year', 'Region'])['CPI Score'].mean().reset_index()

avg_scores_by_year_region['Region'] = avg_scores_by_year_region['Region'].map(region_map)

fig = plt.figure(figsize=(15, 8))

draw_line_plot(x='Year', y='CPI Score', hue='Region', data=avg_scores_by_year_region, title='Corruption by Region')
first_and_last_decade = df[df['Year'].isin([2010, 2019])][['Year', 'Country', 'CPI Score']]

pivoted = first_and_last_decade.pivot_table(values='CPI Score', index='Country', columns='Year').reset_index()

# Yes, we can go for percentage difference rather than absolute, but just keeping things simple

pivoted['Diff'] = pivoted[2019] - pivoted[2010]

gainers = pivoted.nlargest(n=5, columns='Diff')

print(f"Biggest gainers : ")

print(gainers)

losers = pivoted.nsmallest(n=5, columns='Diff')

print(f"Biggest losers : ")

print(losers)
title_gainer_loser = {'Biggest Gainers': gainers, 'Biggest Losers': losers}

for title, gainer_loser in title_gainer_loser.items():

    gainer_loser_df = df[df['Country'].isin(gainer_loser['Country'])]

    draw_line_plot(x='Year', y='CPI Score', hue='Country', data=gainer_loser_df, title=title)
import plotly.express as px

fig = px.choropleth(df[['Year', 'Country', 'CPI Score', 'ISO3']], locations="ISO3", color="CPI Score", 

                    hover_name="Country", hover_data=['CPI Score'], animation_frame="Year",

                    title='Anti-Corruption score over the years (higher is better)')

fig.show()
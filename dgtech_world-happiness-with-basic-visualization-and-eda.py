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
import matplotlib.pyplot as plt

import seaborn as sb

import warnings  

warnings.filterwarnings('ignore')
data_15 = pd.read_csv('/kaggle/input/world-happiness/2015.csv')

data_16 = pd.read_csv('/kaggle/input/world-happiness/2016.csv')

data_17 = pd.read_csv('/kaggle/input/world-happiness/2017.csv')

data_18 = pd.read_csv('/kaggle/input/world-happiness/2018.csv')

data_19 = pd.read_csv('/kaggle/input/world-happiness/2019.csv')

data_19

len(data_19['Country or region'].unique())
data_16['year'] = 2016



cols = data_17.columns

d17 = data_17.rename(columns={cols[1]:'Happiness Rank',cols[2]:'Happiness Score',cols[5]:'Economy (GDP per Capita)',

                      cols[7]:'Health (Life Expectancy)',cols[10]:'Trust (Government Corruption)'

               })

d17['year'] = 2017



cols = data_18.columns

d18 = data_18.rename(columns={cols[0]:'Happiness Rank', cols[1]:'Country', cols[2]:'Happiness Score', cols[3]:'Economy (GDP per Capita)',

                     cols[5]:'Health (Life Expectancy)',cols[6]:'Freedom',cols[8]:'Trust (Government Corruption)'})

d18['year'] = 2018



cols = data_19.columns

d19 = data_19.rename(columns={cols[0]:'Happiness Rank', cols[1]:'Country', cols[2]:'Happiness Score', cols[3]:'Economy (GDP per Capita)',

                     cols[5]:'Health (Life Expectancy)',cols[6]:'Freedom',cols[8]:'Trust (Government Corruption)'})

d19['year'] = 2019
data_19.describe()
plt.figure(figsize=(14,7))



plt.title("Distribution of Happiness Score")

sb.distplot(a=data_19['Score']); # here at the end of line semicolon(;) is for hiding printed object name 
plt.figure(figsize=(14,7))



plt.title("Happiness Score vs GDP per capita")

sb.regplot(data=data_19, x='GDP per capita', y='Score');
plt.figure(figsize=(14,7))

plt.title("Top 10 Countries with High GDP")

sb.barplot(data = data_19.sort_values('GDP per capita', ascending= False).head(10), y='GDP per capita', x='Country or region')

plt.xticks(rotation=90);
plt.figure(figsize= (15,7))



plt.subplot(1,2,1)

plt.title("Perceptions of corruption distribution")

sb.distplot(a=data_19['Perceptions of corruption'], bins =np.arange(0, 0.45+0.2,0.05))

plt.ylabel('Count')



plt.subplot(1,2,2)

plt.title("Happiness Score vs Perceptions of corruption")

sb.regplot(data=data_19, x='Perceptions of corruption', y='Score');

plt.figure(figsize=(14,7))

plt.title("Top 10 Countries with High Perceptions of corruption")

sb.barplot(data = data_19.sort_values('Perceptions of corruption', ascending= False).head(10), x='Country or region', y='Perceptions of corruption')

plt.xticks(rotation=90);
plt.figure(figsize=(14,7))



plt.title("Happiness Score vs Healthy life expectancy")

sb.regplot(data=data_19, x='Healthy life expectancy', y='Score');
plt.figure(figsize=(14,7))

plt.title("Top 10 Countries with High Healthy life expectancy")

sb.barplot(data = data_19.sort_values('Healthy life expectancy', ascending= False).head(10), x='Country or region', y='Perceptions of corruption')

plt.xticks(rotation=90);
plt.figure(figsize=(14,7))



plt.title("Happiness Score vs Social Support")

sb.regplot(data=data_19, x='Social support', y='Score');

plt.figure(figsize=(14,7))

plt.title("Top 10 Countries with Social Support")

sb.barplot(data = data_19.sort_values('Social support', ascending= False).head(10), x='Country or region', y='Social support')

plt.xticks(rotation=90);
plt.figure(figsize=(14,7))



plt.title("Happiness Score vs Freedom to make life choices")

sb.regplot(data=data_19, x='Freedom to make life choices', y='Score');
plt.figure(figsize=(14,7))

plt.title("Top 10 Countries with High Freedom to make life choices")

sb.barplot(data = data_19.sort_values('Freedom to make life choices', ascending= False).head(10), x='Country or region', y='Freedom to make life choices')

plt.xticks(rotation=90);
plt.figure(figsize=(14,7))



plt.title("Happiness Score vs Generosity")

sb.regplot(data=data_19, x='Generosity', y='Score');
plt.figure(figsize=(14,7))



plt.title("Top 10 Countries with High Generosity")

sb.barplot(data = data_19.sort_values('Generosity', ascending= False).head(10), x='Country or region', y='Generosity')

plt.xticks(rotation=90);
p = sb.PairGrid(data_19)

p.map_diag(plt.hist)

p.map_offdiag(plt.scatter);
plt.figure(figsize=(14,7))



plt.title("Correlation Heatmap")

sb.heatmap(data=data_19.corr(), annot=True, vmin=0.005,cmap= 'viridis_r');
plt.figure(figsize=(15,75))

plt.title('Country vs Happiness Score')

sb.barplot(data=data_19.sort_values('Score', ascending=False), x='Score', y='Country or region');
plt.figure(figsize=(14,7))

plt.title("Top 10 Countries with High Happiness Score")

sb.barplot(data = data_19.sort_values('Score', ascending= False).head(10), x='Country or region', y='Score')

plt.xticks(rotation=90);
top_5_country = data_19.sort_values('Score', ascending= False).head(5)['Country or region']

generosity_rank = [np.where(data_19.sort_values('Generosity', ascending= False).reset_index()['Country or region']==i)[0][0] +1 for i in top_5_country]

gdp_rank = [np.where(data_19.sort_values('GDP per capita', ascending= False).reset_index()['Country or region']==i)[0][0] +1 for i in top_5_country]

Social_Support_rank = [np.where(data_19.sort_values('Social support', ascending= False).reset_index()['Country or region']==i)[0][0] +1 for i in top_5_country]

Healthy_life_exp_rank = [np.where(data_19.sort_values('Healthy life expectancy', ascending= False).reset_index()['Country or region']==i)[0][0] +1 for i in top_5_country]

Freedom_choice__rank = [np.where(data_19.sort_values('Freedom to make life choices', ascending= False).reset_index()['Country or region']==i)[0][0] +1 for i in top_5_country]

Perce_corruption_rank = [np.where(data_19.sort_values('Perceptions of corruption', ascending= False).reset_index()['Country or region']==i)[0][0] +1 for i in top_5_country]
feature_rank_top_5_country = pd.DataFrame({

    'country':top_5_country,

    'Generosity_rank':generosity_rank,

    'GDP_rank':gdp_rank,

    'Social_Support_rank':Social_Support_rank,

    'Healthy_life_expectancy_rank':Healthy_life_exp_rank,

    'Freedom_make_choices_rank':Freedom_choice__rank,

    'Perceptions_of_corruption_rank':Perce_corruption_rank})

feature_rank_top_5_country 
plt.figure(figsize=(15,7))

base_color = sb.color_palette()[0]

for i, col in enumerate(feature_rank_top_5_country.columns[1:]):

        plt.subplot(2,3,i+1)

        sb.barplot(data=feature_rank_top_5_country,y=col, x='country', color=base_color)

        plt.xticks(rotation=15)

    
data_16['year'] = 2016



cols = data_17.columns

d17 = data_17.rename(columns={cols[1]:'Happiness Rank',cols[2]:'Happiness Score',cols[5]:'Economy (GDP per Capita)',

                      cols[7]:'Health (Life Expectancy)',cols[10]:'Trust (Government Corruption)'

               })

d17['year'] = 2017



cols = data_18.columns

d18 = data_18.rename(columns={cols[0]:'Happiness Rank', cols[1]:'Country', cols[2]:'Happiness Score', cols[3]:'Economy (GDP per Capita)',

                     cols[5]:'Health (Life Expectancy)',cols[6]:'Freedom',cols[8]:'Trust (Government Corruption)'})

d18['year'] = 2018



cols = data_19.columns

d19 = data_19.rename(columns={cols[0]:'Happiness Rank', cols[1]:'Country', cols[2]:'Happiness Score', cols[3]:'Economy (GDP per Capita)',

                     cols[5]:'Health (Life Expectancy)',cols[6]:'Freedom',cols[8]:'Trust (Government Corruption)'})

d19['year'] = 2019
features = ['year','Country', 'Happiness Rank', 'Happiness Score', 'Economy (GDP per Capita)', 'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)', 'Generosity',]
all_year_data = pd.concat([data_16[features], d17[features], d18[features], d19[features]])
all_year_data
#Reference:- https://plot.ly/python/choropleth-maps/#choropleth-map-with-plotlyexpress

from plotly.offline import iplot,init_notebook_mode

init_notebook_mode(connected=True)

import plotly.express as px

fig = px.choropleth(all_year_data, locations="Country", locationmode='country names',

                     color="Happiness Score",

                     hover_name="Country",

                     animation_frame="year",

                     color_continuous_scale=px.colors.sequential.Plasma)

fig.update_layout(

    title={

        'text': "World Happiness Index 2016-2019",

        'y':0.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})

iplot(fig)



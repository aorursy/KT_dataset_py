import numpy as np

import pandas as pd

import seaborn as sns

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go

from wordcloud import WordCloud

import matplotlib.pyplot as plt

import plotly.express as px

import eli5

from eli5.sklearn import PermutationImportance

from lightgbm import LGBMRegressor



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
d2015 = pd.read_csv("/kaggle/input/world-happiness/2015.csv")

d2016 = pd.read_csv("/kaggle/input/world-happiness/2016.csv")

d2017 = pd.read_csv("/kaggle/input/world-happiness/2017.csv")

d2018 = pd.read_csv("/kaggle/input/world-happiness/2018.csv")

d2019 = pd.read_csv("/kaggle/input/world-happiness/2019.csv")
d2015.drop(["Region",'Standard Error', 'Family', 'Dystopia Residual'],axis=1,inplace=True)

d2015.columns = ["region", "rank", "happiness", "gdp_per_capita",

"healthy_life_expectancy", "freedom_to_life_choise", "corruption_perceptions",

"generosity"]

d2015
d2016.drop(['Region','Lower Confidence Interval','Upper Confidence Interval',

            "Family",'Dystopia Residual'],axis=1,inplace=True)

d2016.columns = ["region","rank","happiness",

                  "gdp_per_capita","healthy_life_expectancy",

                 "freedom_to_life_choise","corruption_perceptions","generosity"]

d2016
d2017.drop(["Whisker.high","Whisker.low",

            "Family","Dystopia.Residual"],axis=1,inplace=True)

d2017.columns =  ["region","rank","happiness",

                  "gdp_per_capita","healthy_life_expectancy",

                 "freedom_to_life_choise","generosity","corruption_perceptions"]

d2017
d2018.columns = ["rank","region","happiness",

                  "gdp_per_capita","social_support","healthy_life_expectancy",

                 "freedom_to_life_choise","generosity","corruption_perceptions"]

pd.set_option('display.width', 500)

pd.set_option('display.expand_frame_repr', False)

d2018
coltoselect = ["rank","region","happiness",

                "gdp_per_capita","healthy_life_expectancy",

                "freedom_to_life_choise","generosity","corruption_perceptions"]

d2019.columns = ["rank","region","happiness",

                  "gdp_per_capita","social_support","healthy_life_expectancy",

                 "freedom_to_life_choise","generosity","corruption_perceptions"]

d2019
# Correlation in dataset 2015

d2015.corr()
# Correlation in dataset 2016

d2016.corr()
# Correlation in dataset 2017

d2017.corr()
# Correlation in dataset 2018

d2018.corr()
# Correlation in dataset 2019

d2019.corr()
d2015["year"] = str(2015)

d2016["year"] = str(2016)

d2017["year"] = str(2017)

d2018["year"] = str(2018)

d2019["year"] = str(2019)
finaldf = d2015.append([d2016,d2017,d2018,d2019])

finaldf.head()
d2015.sort_values("gdp_per_capita",inplace=True)

d2016.sort_values("gdp_per_capita",inplace=True)

d2017.sort_values("gdp_per_capita",inplace=True)

d2018.sort_values("gdp_per_capita",inplace=True)

d2019.sort_values("gdp_per_capita",inplace=True)
fig = px.scatter(finaldf, x="gdp_per_capita", 

                 y="happiness",

                 facet_row="year",

                color="year",

                trendline= "ols")

fig.update(layout_coloraxis_showscale=False)

fig.update_traces(textposition='top center')

fig.update_layout(

    height=800,

    title_text='GDP per capita and Happiness Score'

)

fig.show()
fig = px.scatter(finaldf, x="healthy_life_expectancy", 

                 y="happiness",

                 facet_row="year",

                color="year",

                trendline= "ols")

fig.update(layout_coloraxis_showscale=False)

fig.update_traces(textposition='top center')

fig.update_layout(

    height=800,

    title_text='Healthy Life Expecancy and Happiness Score'

)

fig.show()
fig = px.scatter(finaldf, x="freedom_to_life_choise", 

                 y="happiness",

                 facet_row="year",

                color="year",

                trendline= "ols")

fig.update(layout_coloraxis_showscale=False)

fig.update_traces(textposition='top center')

fig.update_layout(

    height=800,

    title_text='Freedom to Life Choises and Happiness Score'

)

fig.show()
fig = px.scatter(finaldf, x="generosity", 

                 y="happiness",

                 facet_row="year",

                color="year",

                trendline= "ols")

fig.update(layout_coloraxis_showscale=False)

fig.update_traces(textposition='top center')

fig.update_layout(

    height=800,

    title_text='Generosity and Happiness Score'

)

fig.show()
fig = px.scatter(finaldf, x="corruption_perceptions", 

                 y="happiness",

                 facet_row="year",

                color="year",

                trendline= "ols")

fig.update(layout_coloraxis_showscale=False)

fig.update_traces(textposition='top center')

fig.update_layout(

    height=800,

    title_text='Perception about corruption of Goverment and Happiness Score'

)

fig.show()
fig = px.scatter(finaldf, x="gdp_per_capita", y="happiness", animation_frame="year",

           animation_group="region",

           size="rank", color="region", hover_name="region",

          trendline= "ols")



fig.update_layout(

    title_text='Happiness Score vs GDP per Capita'

)

fig.show()
fig = px.scatter(finaldf, x="healthy_life_expectancy", y="happiness", animation_frame="year",

           animation_group="region",

           size="rank", color="region", hover_name="region")



fig.update_layout(

    title_text='Happiness Score vs Healthy Life Expectancy'

)

fig.show()
fig = px.scatter(finaldf, x="freedom_to_life_choise", y="happiness", animation_frame="year",

           animation_group="region",

           size="rank", color="region", hover_name="region")



fig.update_layout(

    title_text='Happiness Score vs Freedom to Life choises'

)

fig.show()
fig = px.scatter(finaldf, x="generosity", y="happiness", animation_frame="year",

           animation_group="region",

           size="rank", color="region", hover_name="region")



fig.update_layout(

    title_text='Happiness Score vs Generosity'

)

fig.show()
fig = px.scatter(finaldf, x="corruption_perceptions", y="happiness", animation_frame="year",

           animation_group="region",

           size="rank", color="region", hover_name="region")



fig.update_layout(

    title_text='Happiness Score vs Corruption Perceptions'

)

fig.show()
import plotly.graph_objs as go

from plotly.offline import iplot



data = dict(type = 'choropleth', 

           locations = d2015['region'],

           locationmode = 'country names',

           colorscale='RdYlGn',

           z = d2015['happiness'], 

           text = d2015['region'],

           colorbar = {'title':'Happiness'})



layout = dict(title = 'Geographical Visualization of Happiness Score in 2015', 

              geo = dict(showframe = True, projection = {'type': 'azimuthal equal area'}))



choromap3 = go.Figure(data = [data], layout=layout)

iplot(choromap3)
data = dict(type = 'choropleth', 

           locations = d2016['region'],

           locationmode = 'country names',

           colorscale='RdYlGn',

           z = d2016['happiness'], 

           text = d2016['region'],

           colorbar = {'title':'Happiness'})



layout = dict(title = 'Geographical Visualization of Happiness Score in 2016', 

              geo = dict(showframe = True, projection = {'type': 'azimuthal equal area'}))



choromap3 = go.Figure(data = [data], layout=layout)

iplot(choromap3)
data = dict(type = 'choropleth', 

           locations = d2017['region'],

           locationmode = 'country names',

           colorscale='RdYlGn',

           z = d2017['happiness'], 

           text = d2017['region'],

           colorbar = {'title':'Happiness'})



layout = dict(title = 'Geographical Visualization of Happiness Score in 2017', 

              geo = dict(showframe = True, projection = {'type': 'azimuthal equal area'}))



choromap3 = go.Figure(data = [data], layout=layout)

iplot(choromap3)
data = dict(type = 'choropleth', 

           locations = d2018['region'],

           locationmode = 'country names',

           colorscale='RdYlGn',

           z = d2018['happiness'], 

           text = d2018['region'],

           colorbar = {'title':'Happiness'})



layout = dict(title = 'Geographical Visualization of Happiness Score in 2018', 

              geo = dict(showframe = True, projection = {'type': 'azimuthal equal area'}))



choromap3 = go.Figure(data = [data], layout=layout)

iplot(choromap3)
data = dict(type = 'choropleth', 

           locations = d2019['region'],

           locationmode = 'country names',

           colorscale='RdYlGn',

           z = d2019['happiness'], 

           text = d2019['region'],

           colorbar = {'title':'Happiness'})



layout = dict(title = 'Geographical Visualization of Happiness Score in 2019', 

              geo = dict(showframe = True, projection = {'type': 'azimuthal equal area'}))



choromap3 = go.Figure(data = [data], layout=layout)

iplot(choromap3)
plt.figure(figsize=(14,8))

df = finaldf[finaldf['region']=='Egypt']

sns.lineplot(x="year", y="gdp_per_capita",data=df,label='Egypt')



df = finaldf[finaldf['region']=='United States']

sns.lineplot(x="year", y="gdp_per_capita",data=df,label='US')



df = finaldf[finaldf['region']=='Jordan']

sns.lineplot(x="year", y="gdp_per_capita",data=df,label='Jordan')



df = finaldf[finaldf['region']=='United Kingdom']

sns.lineplot(x="year", y="gdp_per_capita",data=df,label="UK")





plt.title("GDP per capita comparison 2015-2019")
plt.figure(figsize=(14,8))

df = finaldf[finaldf['region']=='Egypt']

sns.lineplot(x="year", y="healthy_life_expectancy",data=df,label='Egypt')



df = finaldf[finaldf['region']=='United States']

sns.lineplot(x="year", y="healthy_life_expectancy",data=df,label='US')



df = finaldf[finaldf['region']=='Jordan']

sns.lineplot(x="year", y="healthy_life_expectancy",data=df,label='Jordan')



df = finaldf[finaldf['region']=='United Kingdom']

sns.lineplot(x="year", y="healthy_life_expectancy",data=df,label="UK")





plt.title("healthy life expectancy comparison 2015-2019")
plt.figure(figsize=(14,8))

df = finaldf[finaldf['region']=='Egypt']

sns.lineplot(x="year", y="freedom_to_life_choise",data=df,label='Egypt')



df = finaldf[finaldf['region']=='United States']

sns.lineplot(x="year", y="freedom_to_life_choise",data=df,label='US')



df = finaldf[finaldf['region']=='Jordan']

sns.lineplot(x="year", y="freedom_to_life_choise",data=df,label='Jordan')



df = finaldf[finaldf['region']=='United Kingdom']

sns.lineplot(x="year", y="freedom_to_life_choise",data=df,label="UK")





plt.title("freedom to life choise comparison 2015-2019")
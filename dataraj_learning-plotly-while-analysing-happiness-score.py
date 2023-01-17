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
d2015 = pd.read_csv("/kaggle/input/world-happiness/2015.csv")

d2016 = pd.read_csv("/kaggle/input/world-happiness/2016.csv")

d2017 = pd.read_csv("/kaggle/input/world-happiness/2017.csv")

d2018 = pd.read_csv("/kaggle/input/world-happiness/2018.csv")

d2019 = pd.read_csv("/kaggle/input/world-happiness/2019.csv")
coltoselect = ["rank","region","score",

                "gdp_per_capita","healthy_life_expectancy",

                "freedom_to_life_choise","generosity","corruption_perceptions"]

d2019.columns = ["rank","region","score",

                  "gdp_per_capita","social_support","healthy_life_expectancy",

                 "freedom_to_life_choise","generosity","corruption_perceptions"]

d2019.head()
d2018.columns = ["rank","region","score",

                  "gdp_per_capita","social_support","healthy_life_expectancy",

                 "freedom_to_life_choise","generosity","corruption_perceptions"]

pd.set_option('display.width', 500)

pd.set_option('display.expand_frame_repr', False)

d2018.head()
d2017.drop(["Whisker.high","Whisker.low",

            "Family","Dystopia.Residual"],axis=1,inplace=True)

d2017.columns =  ["region","rank","score",

                  "gdp_per_capita","healthy_life_expectancy",

                 "freedom_to_life_choise","generosity","corruption_perceptions"]

d2017.head()
d2016.drop(['Region','Lower Confidence Interval','Upper Confidence Interval',

            "Family",'Dystopia Residual'],axis=1,inplace=True)

d2016.columns = ["region","rank","score",

                  "gdp_per_capita","healthy_life_expectancy",

                 "freedom_to_life_choise","corruption_perceptions","generosity"]

d2016.head()
d2015.drop(["Region",'Standard Error', 'Family', 'Dystopia Residual'],axis=1,inplace=True)

d2015.columns = ["region", "rank", "score", "gdp_per_capita",

"healthy_life_expectancy", "freedom_to_life_choise", "corruption_perceptions",

"generosity"]

d2015.head()
d2015 = d2015.loc[:,coltoselect].copy()

d2016 = d2016.loc[:,coltoselect].copy()

d2017 = d2017.loc[:,coltoselect].copy()

d2018 = d2018.loc[:,coltoselect].copy()

d2019 = d2019.loc[:,coltoselect].copy()
d2015["year"] = 2015

d2016["year"] = 2016

d2017["year"] = 2017

d2018["year"] = 2018

d2019["year"] = 2019
finaldf = d2015.append([d2016,d2017,d2018,d2019])

finaldf.head()
d2015.sort_values("gdp_per_capita",inplace=True)

d2016.sort_values("gdp_per_capita",inplace=True)

d2017.sort_values("gdp_per_capita",inplace=True)

d2018.sort_values("gdp_per_capita",inplace=True)

d2019.sort_values("gdp_per_capita",inplace=True)
import missingno as msno

msno.bar(finaldf)

plt.show()
msno.matrix(finaldf)

plt.show()
finaldf.loc[finaldf.isnull().any(axis=1),:]
finaldf.dropna(inplace=True)
p15 = go.Scatter(

                    x = d2015.gdp_per_capita,

                    y = d2015.score,

                    mode = "lines",

                    name = "2015",

                    marker = dict(color = 'green'),

                    text= d2015.region)

p16 = go.Scatter(

                    x = d2016.gdp_per_capita,

                    y = d2016.score,

                    mode = "lines",

                    name = "2016",

                    marker = dict(color = 'red'),

                    text= d2016.region)



p17 = go.Scatter(

                    x = d2017.gdp_per_capita,

                    y = d2017.score,

                    mode = "lines",

                    name = "2017",

                    marker = dict(color = 'violet'),

                    text= d2017.region)



p18 = go.Scatter(

                    x = d2018.gdp_per_capita,

                    y = d2018.score,

                    mode = "lines",

                    name = "2018",

                    marker = dict(color = 'blue'),

                    text= d2018.region)



p19 = go.Scatter(

                    x = d2019.gdp_per_capita,

                    y = d2019.score,

                    mode = "lines",

                    name = "2019",

                    marker = dict(color = 'black'),

                    text= d2019.region)





data = [p15, p16, p17, p18, p19]

properties = dict(title = 'Happiness Score vs GDP per Capita',

              xaxis= dict(title= 'GDP per Capita',ticklen= 5,zeroline= False),

             yaxis= dict(title= 'Happiness Score',ticklen= 5,zeroline= False),

             )

fig = dict(data = data, layout = properties)

iplot(fig)
fig = px.scatter(finaldf, x="gdp_per_capita", 

                 y="score",

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

                 y="score",

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

                 y="score",

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

                 y="score",

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

                 y="score",

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

finaldf.head()
px.scatter(finaldf, x="gdp_per_capita", y="score", animation_frame="year",

           animation_group="region",

           size="rank", color="region", hover_name="region",

          trendline= "ols")
px.scatter(finaldf, x="healthy_life_expectancy", y="score", animation_frame="year",

           animation_group="region",

           size="rank", color="region", hover_name="region")
px.scatter(finaldf, x="freedom_to_life_choise", y="score", animation_frame="year",

           animation_group="region",

           size="rank", color="region", hover_name="region")
px.scatter(finaldf, x="generosity", y="score", animation_frame="year",

           animation_group="region",

           size="rank", color="region", hover_name="region")
px.scatter(finaldf, x="corruption_perceptions", y="score", animation_frame="year",

           animation_group="region",

           size="rank", color="region", hover_name="region")
lgbm = LGBMRegressor(n_estimators=5000)

indData = finaldf.loc[:,"gdp_per_capita":"year"]

depData = finaldf.pop("score")

lgbm.fit(indData, depData)

columns = indData.columns.to_list()

perm = PermutationImportance(lgbm, random_state=10).fit(indData, depData)

eli5.show_weights(perm, feature_names = columns)
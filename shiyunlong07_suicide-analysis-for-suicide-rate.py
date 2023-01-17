# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path = "/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv"

df = pd.read_csv(path)

df.head()
df.info()
import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go



%matplotlib inline

sns.set_style("darkgrid")
# 2.1 Let's check the information on different countries

nb_unique_country = len(df['country'].unique())

print("There are %i countries in the dataframe."%nb_unique_country)
# Visualization top 20 countries suicide data

df_country = df.groupby(by = ['country']).agg({'suicides_no':['sum']})

df_country.columns = ['total_suicide']

df_country.reset_index(inplace = True)

df_country = df_country.sort_values(by = ['total_suicide'], ascending = False).head(20)



fig = px.bar(df_country, x = 'country', y = 'total_suicide',title = "Top 20 Total Suicide Country")

fig.show()
labels = df_country["country"]

values = df_country["total_suicide"]

fig = go.Figure(data = [go.Pie(labels = labels, values = values)])

fig.update_layout(title_text = 'Suicide Number Distribution per Countries')

fig.show()
country = ["Russian Federation","United States","Japan","France"]



for i,column in enumerate(country):

    df_tmp = df[df['country'] == column]

    df_cy = df_tmp.groupby(by=['year']).agg({'suicides_no':['sum']})

    df_cy.columns = ['total_suicide']

    df_cy.reset_index(inplace=True)
# Russian Federation data

df_rf = df[df['country']=='Russian Federation']

df_rf_y=df_rf.groupby(by=['year']).agg({'suicides_no':['sum']})

df_rf_y.columns = ['total_suicide']

df_rf_y.reset_index(inplace = True)



# US data

df_us = df[df['country']=='United States']

df_us_y=df_us.groupby(by=['year']).agg({'suicides_no':['sum']})

df_us_y.columns = ['total_suicide']

df_us_y.reset_index(inplace = True)



#Japan data

df_jp = df[df['country']=='Japan']

df_jp_y=df_jp.groupby(by=['year']).agg({'suicides_no':['sum']})

df_jp_y.columns = ['total_suicide']

df_jp_y.reset_index(inplace = True)



#France data

df_fr = df[df['country']=='France']

df_fr_y=df_fr.groupby(by=['year']).agg({'suicides_no':['sum']})

df_fr_y.columns = ['total_suicide']

df_fr_y.reset_index(inplace = True)



fig = go.Figure()



#Russian Federation trace

fig.add_trace(go.Bar(x = df_rf_y['year'],

                    y = df_rf_y['total_suicide'],

                    name = 'Russian Federation',

                    marker_color='rgb(55,83,109)'

                    ))



#US trace

fig.add_trace(go.Bar(x = df_us_y['year'],

                    y = df_us_y['total_suicide'],

                    name = 'United States',

                    marker_color='rgb(26,118,255)'

                    ))



#Japan trace

fig.add_trace(go.Bar(x = df_jp_y['year'],

                    y = df_jp_y['total_suicide'],

                    name = 'Japan',

                    marker_color='rgb(255,128,0)'

                    ))



#France trace

fig.add_trace(go.Bar(x = df_fr_y['year'],

                    y = df_fr_y['total_suicide'],

                    name = 'France',

                    marker_color='rgb(99,225,45)'

                    ))



fig.update_layout(

    title='Top 4 Countries Total Suicide per Year',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title='Total Suicide Number',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.15,

    bargroupgap=0.1

)



fig.show()
df_gdp = df.groupby(by=['country', 'year', 'sex', 'gdp_per_capita ($)']).agg({"suicides_no": ['sum']})

df_gdp.columns = ['total_suicide']

df_gdp.reset_index(inplace = True)



#Plot the relationship between GDP per capital and suicide number (seperate by sex)

fig = px.scatter(df_gdp, 

                 x = 'gdp_per_capita ($)', 

                 y = 'total_suicide',

                 color = 'sex'

                )



fig.update_layout(title='Relation between Suicide Number and GDP per Capita')



fig.show()
df_gdp.head()
df_gdp_rf = df_gdp[df_gdp['country']=='Russian Federation']

df_gdp_us = df_gdp[df_gdp['country']=='United States']

df_gdp_jp = df_gdp[df_gdp['country']=='Japan']

df_gdp_fr = df_gdp[df_gdp['country']=='France']

#df_gdp_rf.head()

fig = go.Figure()



#Russian Federation plot

fig.add_trace(go.Scatter(x =df_gdp_rf['gdp_per_capita ($)'],

                        y = df_gdp_rf['total_suicide'],

                        mode = 'markers',

                        name = 'Russian Federation'))



fig.add_trace(go.Scatter(x =df_gdp_us['gdp_per_capita ($)'],

                        y = df_gdp_us['total_suicide'],

                        mode = 'markers',

                        name = 'United States'))



fig.add_trace(go.Scatter(x =df_gdp_jp['gdp_per_capita ($)'],

                        y = df_gdp_jp['total_suicide'],

                        mode = 'markers',

                        name = 'Japan'))



fig.add_trace(go.Scatter(x =df_gdp_fr['gdp_per_capita ($)'],

                        y = df_gdp_fr['total_suicide'],

                        mode = 'markers',

                        name = 'France'))







fig.update_layout(title ='Top 4 Countries Suicide by GDP per Capita',

                 xaxis = dict(title = 'GDP per Capital ($)'),

                 yaxis = dict(title = 'Total Suicide Number'))



fig.show()
df_age = df.groupby(by = ['age', 'sex']).agg({'population':['sum'], 'suicides_no':['sum']})

df_age.columns = ['total_population','total_suicide']

df_age.reset_index(inplace = True)

#df_age.head()



labels = df_age["age"]

values = df_age["total_population"]

fig = go.Figure(data = [go.Pie(labels = labels, 

                               values = values)])

fig.update_layout(title_text = 'Suicide Age Distribution')

fig.show()
labels = df_age["sex"]

values = df_age["total_suicide"]

fig = go.Figure(data = [go.Pie(labels = labels, 

                               values = values)])

fig.update_layout(title_text = 'Suicide Sex Distribution')

fig.show()
fig = px.bar(df_age, 

                 x = 'age', 

                 y = 'total_suicide',

                 color = 'sex'

                )



fig.update_layout(title='Suicide Distribution by age/sex')



fig.show()
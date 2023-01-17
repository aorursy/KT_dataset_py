# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

# word cloud library

from wordcloud import WordCloud



# matplotlib

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataCountries=pd.read_csv('../input/countries-of-the-world/countries of the world.csv')
dataCountries.head()
dataCountries.info()
regions=dataCountries.Region.unique()

dataCountries['Pop. Density (per sq. mi.)']=([float(each.replace(',','.')) for each in dataCountries['Pop. Density (per sq. mi.)']])

population=[]

pop_density=[]

for i in regions:

    x=dataCountries[dataCountries.Region==i]

    population.append(sum(x.Population))

    pop_density.append(sum(x['Pop. Density (per sq. mi.)']))

pop_reg=pd.DataFrame({"Regions":regions,

                    "Population":population,

                    "PopulationDensity":pop_density})

pop_reg.sort_values(by=['Population'],inplace=True,ascending=False)

population_bar=go.Bar(x=pop_reg.Regions,

                      y=pop_reg.Population,

                     name="Population",

                     marker = dict(color = 'rgba(50, 134, 55, 0.5)',

                     line=dict(color='rgb(0,0,0)',width=1.5)),

                     text=pop_reg.Regions,

                     )

layout=go.Layout(title='Population by Regions',)

fig=go.Figure(data=population_bar,layout=layout)

iplot(fig)
pop_den=pop_reg.sort_values(by=['PopulationDensity'],ascending=False)

pop_density_bar=go.Bar(x=pop_den.Regions,

                      y=pop_den.PopulationDensity,

                     name="Population Density",

                     marker = dict(color = 'rgba(50, 134, 55, 0.5)',

                     line=dict(color='rgb(0,0,0)',width=1.5)),

                     text=pop_den.Regions,

                     )

layout=go.Layout(title='Population Density by Regions (per sq. mi.)',)

fig=go.Figure(data=pop_density_bar,layout=layout)

iplot(fig)
population_pie={

  "data": [

    {

      "values": pop_reg.Population,

      "labels": pop_reg.Regions,

      "domain": {"x": [0, .8],

                "y": [0, .8]},

      "name": "Population Rate of Regions",

      "hoverinfo":"label+percent+name",

      "hole": .2,

      "type": "pie"

    },],

  "layout": {

        "title":"Population Rate of Regions",

        "annotations": [

            { "font": { "size": 20},

              "showarrow": False,

              "text": "Population Rate of Regions",

                "x": 0.27,

                "y": 1

            },

        ]

    }

}

    

iplot(population_pie)



pop= go.Scatter(

                    x = dataCountries.Country,

                    y = dataCountries.Population,

                    mode = "lines+markers",

                    name = "Population",

                    marker = dict(color = 'rgba(80, 26, 255, 0.8)'),

                    text= dataCountries.Country)

data = [pop]

layout = dict(title = 'Population by Country',

              xaxis= dict(title= 'Country',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
data_pop=dataCountries.loc[:,['Country','Population']]

data_pop.sort_values(by=['Population'],inplace=True,ascending=False)

pop_bar=px.bar(data_pop[:10],x='Country',y='Population',title="Top 10 Countries with the Most Population")

pop_bar.show()
pop_den_con= go.Scatter(

                    x = dataCountries.Country,

                    y = dataCountries['Pop. Density (per sq. mi.)'],

                    mode = "lines+markers",

                    name = "Population Density",

                    marker = dict(color = 'rgba(80, 26, 255, 0.8)'),

                    text= dataCountries.Country)

data = [pop_den_con]

layout = dict(title = 'Population Density (per sq. mi.)  by Country',

              xaxis= dict(title= 'Country',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
data_infant_morality=dataCountries.loc[:,['Country','Infant mortality (per 1000 births)']]

data_infant_morality.dropna(inplace=True)

data_infant_morality['Infant mortality (per 1000 births)']=[float(each.replace(',','.')) for each in data_infant_morality['Infant mortality (per 1000 births)']]
infantMortality= go.Scatter(

                    x = data_infant_morality.Country,

                    y = data_infant_morality['Infant mortality (per 1000 births)'],

                    mode = "lines+markers",

                    name = "Infant Mortality Rates",

                    marker = dict(color = 'rgba(80, 26, 255, 0.8)'),

                    text= data_infant_morality.Country)

data = [infantMortality]

layout = dict(title = 'Infant Mortality Rates by Country (per 1000 births)',

              xaxis= dict(title= 'Country',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
data_infant_morality.sort_values(by=['Infant mortality (per 1000 births)'],inplace=True,ascending=False)

inf_mor_bar=px.bar(data_infant_morality[:10],x='Country',y='Infant mortality (per 1000 births)',title="Top 10 Countries with Highest Infant Mortality Rates")

inf_mor_bar.show()
birthrate = go.Scatter(

    x = dataCountries.index,

    y = dataCountries.Birthrate,

    mode = 'lines+markers',

    name = 'Birthrate',

    marker = dict(color = 'rgba(10, 255, 10, 0.5)'),

    text = dataCountries.Country)



deathrate = go.Scatter(

    x = dataCountries.index,

    y = dataCountries.Deathrate,

    mode = 'lines+markers',

    name = 'Deathrate',

    marker = dict(color = 'rgba(255, 10, 10, 0.5)'),

    text = dataCountries.Country)



layout = dict(title = 'Birth and Death Rate of Countries',

             xaxis= dict(zeroline= False)

             )



data = [birthrate, deathrate]

fig = dict(data = data, layout = layout)



iplot(fig)


literacy=dataCountries.loc[:,['Country','Literacy (%)']]

literacy.dropna(inplace=True)

literacy['Literacy (%)']=([float(each.replace(',','.')) for each in literacy['Literacy (%)']])

literacy.columns=['Country','Literacy']

highest_literacy=literacy.sort_values(by='Literacy',ascending=False)[:10]
literacy_of_countries = highest_literacy.Country

plt.subplots(figsize=(8,8))

literacy_wordcloud = WordCloud(

                          background_color='white',

                          width=512,

                          height=384

                         ).generate(" ".join(literacy_of_countries))

plt.imshow(literacy_wordcloud)

plt.axis('off')

plt.savefig('graph.png')



plt.show()


dataSector=dataCountries.loc[:,['Country','Agriculture','Industry','Service','GDP ($ per capita)']]

dataSector.dropna(inplace=True)

dataSector.Agriculture=[float(each.replace(',','.')) for each in dataSector.Agriculture]

dataSector.Industry=[float(each.replace(',','.')) for each in dataSector.Industry]

dataSector.Service=[float(each.replace(',','.')) for each in dataSector.Service]



sector_scatter = px.scatter_3d(dataSector, x='Agriculture', y='Industry', z='Service',

                    color='Country',

                    title='Sectoral Distribution',  

                   )

sector_scatter.show()
#editting data

agr=list(dataSector.Agriculture)

ind=list(dataSector.Industry)

serv=list(dataSector.Service)

gdp=list(dataSector['GDP ($ per capita)'])

leading_sector=[]

for i in range(211):

    x=max(agr[i],ind[i],serv[i])

    if x== agr[i]:

        leading_sector.append('Agriculture')

    elif x==ind[i]:

        leading_sector.append('Industry')

    elif x==serv[i]:

        leading_sector.append('Service')

dataSector2=pd.DataFrame({"Country":dataSector.Country,

                         "Agriculture":agr,

                         "Industry":ind,

                         "Service":serv,

                         "LeadingSector":leading_sector,

                         "GDP":gdp})

fig = px.scatter(dataSector2, x="Country", y="GDP",color="LeadingSector")

fig.show()

df_arable=dataCountries.loc[:,['Country','Arable (%)']]

df_arable.dropna(inplace=True)

df_arable['Arable (%)']=([float(each.replace(',','.')) for each in df_arable['Arable (%)']])

df_arable=df_arable.sort_values(by='Arable (%)',ascending=False)



arable_bar=px.bar(df_arable[:10],x='Country',y='Arable (%)')

arable_bar.show()

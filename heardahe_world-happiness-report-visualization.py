import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))



import seaborn as sns

%matplotlib inline



import matplotlib.pyplot as plt

from pandas.tools.plotting import parallel_coordinates



import plotly.plotly as py

import plotly.graph_objs as go 

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
df15 = pd.read_csv("../input/2015.csv")

df16 = pd.read_csv("../input/2016.csv")

df17 = pd.read_csv("../input/2017.csv")
ColumnNames = pd.DataFrame([df15.columns.values,df16.columns.values,df17.columns.values]).T



ColumnNames.columns = ['2015','2016','2017']

                                     

ColumnNames
df15.describe()
df15.info()
df15.head(5)
df15.tail(5)
sns.jointplot(x='Economy (GDP per Capita)',y='Happiness Score',data=df15,kind='scatter')
sns.jointplot(x='Health (Life Expectancy)',y='Happiness Score',data=df15,kind='scatter')
sns.pairplot(df15)
sns.heatmap(df15.corr())
df16.describe()
df16.info()
df16.head(5)
df16.tail(5)
sns.jointplot(x='Economy (GDP per Capita)',y='Happiness Score',data=df16,kind='scatter')
sns.jointplot(x='Health (Life Expectancy)',y='Happiness Score',data=df16,kind='scatter')
sns.pairplot(df16)
sns.heatmap(df16.corr())
df17.describe()
df17.info()
df17.head(5)
df17.tail(5)
sns.jointplot(x='Economy..GDP.per.Capita.',y='Happiness.Score',data=df17,kind='scatter')
sns.jointplot(x='Health..Life.Expectancy.',y='Happiness.Score',data=df17,kind='scatter')
sns.pairplot(df17)
sns.heatmap(df17.corr())
#Top Ten countries' names for ever year.

TopTen2015 = df15.head(10).filter(['Country'])

TopTen2016 = df16.head(10).filter(['Country']) 

TopTen2017 = df17.head(10).filter(['Country'])



#Merge,removing duplicates and sort by name.

TTC = pd.concat([TopTen2015['Country'],TopTen2016['Country'],TopTen2017['Country']])

TTC = TTC.drop_duplicates()

TTC = pd.DataFrame(TTC)

TTC = TTC.sort_values(by=['Country'])



#No need to use TopTen2015,TopTen2016,TopTen2017 variables.

del TopTen2015 

del TopTen2016 

del TopTen2017





#Merging with Happiness Rank and Happiness Score by country name and rename columns for each year.

TTC = TTC.merge(df15,on='Country').filter(['Country','Happiness Rank','Happiness Score'])

TTC = TTC.rename(columns = {'Happiness Rank': 'Happiness Rank 2015',

                                          'Happiness Score' : 'Happiness Score 2015'})



TTC = TTC.merge(df16,on='Country').filter(['Country','Happiness Rank','Happiness Score',

                                                       'Happiness Rank 2015','Happiness Score 2015'])

TTC = TTC.rename(columns = {'Happiness Rank': 'Happiness Rank 2016',

                                          'Happiness Score' : 'Happiness Score 2016'})



TTC = TTC.merge(df17,on='Country').filter(['Country','Happiness.Rank','Happiness.Score',

                                                      'Happiness Rank 2015','Happiness Score 2015',

                                                      'Happiness Rank 2016','Happiness Score 2016'])

TTC = TTC.rename(columns = {'Happiness.Rank': 'Happiness Rank 2017',

                                          'Happiness.Score' : 'Happiness Score 2017'})



#Order positions of columns by year.

TTC = TTC [['Country','Happiness Rank 2015','Happiness Score 2015',

              'Happiness Rank 2016','Happiness Score 2016',

              'Happiness Rank 2017','Happiness Score 2017']]



TTC
ParallelPlot = TTC.filter(['Country','Happiness Rank 2015','Happiness Rank 2016','Happiness Rank 2017'])

plt1 = parallel_coordinates(ParallelPlot,'Country',colormap='Set1')

plt1.legend(loc='upper center', bbox_to_anchor=(1.4,1), shadow=True, ncol=1)
ParallelPlot2 = TTC.filter(['Country','Happiness Score 2015','Happiness Score 2016','Happiness Score 2017'])

plt2 = parallel_coordinates(ParallelPlot2,'Country',colormap='Set1')

plt2.legend(loc='upper center', bbox_to_anchor=(1.4,1), shadow=True, ncol=1)
data = dict(

        type = 'choropleth',

        locations = df15['Country'],

        locationmode = 'country names',

        z = df15['Happiness Score'],

        text = df15['Country'],

        colorbar = {'title' : 'Happiness Score'},

      )



layout = dict(

            title = '2015',

            geo = dict(

            showframe = False,

                projection = {'type':'mercator'}

        )

    )



choromap15 = go.Figure(data = [data],layout = layout)

iplot(choromap15)
data = dict(

        type = 'choropleth',

        locations = df16['Country'],

        locationmode = 'country names',

        z = df16['Happiness Score'],

        text = df16['Country'],

        colorbar = {'title' : 'Happiness Score'},

      )



layout = dict(

            title = '2016',

            geo = dict(

            showframe = False,

                projection = {'type':'mercator'}

        )

    )



choromap16 = go.Figure(data = [data],layout = layout)

iplot(choromap16)
data = dict(

        type = 'choropleth',

        locations = df17['Country'],

        locationmode = 'country names',

        z = df17['Happiness.Score'],

        text = df17['Country'],

        colorbar = {'title' : 'Happiness Score'},

      )



layout = dict(

            title = '2017',

            geo = dict(

            showframe = False,

                projection = {'type':'mercator'}

        )

    )



choromap17 = go.Figure(data = [data],layout = layout)

iplot(choromap17)
data15 = [ go.Scattergeo(

        locations = df15['Country'],

        locationmode = 'country names',

        mode = 'markers',

        marker = dict( 

            size = 8, 

            opacity = 0.8,

            reversescale = False,

            autocolorscale = True,

            symbol = 'circle',

            line = dict(

                width=1,

                color='rgba(102, 102, 102)'

            ),

            

            cmin = df15['Economy (GDP per Capita)'].min(),

            color = df15['Economy (GDP per Capita)'],

            cmax = df15['Economy (GDP per Capita)'].max(),

            colorbar=dict(

                title="GDP per Capita"

            )

        ))]



layout15 = dict(

        title = 'GDP per Capita - 2015', 

        geo = dict(

            scope='world',

            projection=dict(type='natural earth'),

            showland = False,

            ),

    )



fig15 = go.Figure(data=data15, layout=layout15 )

iplot(fig15)
data16 = [ go.Scattergeo(

        locations = df16['Country'],

        locationmode = 'country names',

        mode = 'markers',

        marker = dict( 

            size = 8, 

            opacity = 0.8,

            reversescale = False,

            autocolorscale = True,

            symbol = 'circle',

            line = dict(

                width=1,

                color='rgba(102, 102, 102)'

            ),

            

            cmin = df16['Economy (GDP per Capita)'].min(),

            color = df16['Economy (GDP per Capita)'],

            cmax = df16['Economy (GDP per Capita)'].max(),

            colorbar=dict(

                title="GDP per Capita"

            )

        ))]



layout16 = dict(

        title = 'GDP per Capita - 2016', 

        geo = dict(

            scope='world',

            projection=dict(type='natural earth'),

            showland = False,

            ),

    )



fig16 = go.Figure(data=data16, layout=layout16 )

iplot(fig16)
data17 = [ go.Scattergeo(

        locations = df17['Country'],

        locationmode = 'country names',

        mode = 'markers',

        marker = dict( 

            size = 8, 

            opacity = 0.8,

            reversescale = False,

            autocolorscale = True,

            symbol = 'circle',

            line = dict(

                width=1,

                color='rgba(102, 102, 102)'

            ),

            

            cmin = df17['Economy..GDP.per.Capita.'].min(),

            color = df17['Economy..GDP.per.Capita.'],

            cmax = df17['Economy..GDP.per.Capita.'].max(),

            colorbar=dict(

                title="GDP per Capita"

            )

        ))]



layout17 = dict(

        title = 'GDP per Capita - 2017)', 

        geo = dict(

            scope='world',

            projection=dict(type='natural earth'),

            showland = False,

            ),

    )



fig17 = go.Figure(data=data17, layout=layout17 )

iplot(fig17)
#Import relevant libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import squarify

import matplotlib.ticker as plticker

from matplotlib.ticker import MultipleLocator, FormatStrFormatter

plt.style.use('fivethirtyeight')

import warnings

warnings.filterwarnings('ignore')

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import base64

import io

import codecs

from IPython.display import HTML

import jupyter
data=pd.read_csv('../input/WorldPopulation.csv',encoding='ISO-8859-1')

data.head(10)

#data.info()
index_min = np.argmin(data['2016'])

index_max = np.argmax(data['2016'])

unit_min = data['Country'].values[index_min]

unit_max = data['Country'].values[index_max]



print('The most populated political unit:',  unit_max, '-', 

round(data['2016'].max()),'; The least populated:', unit_min, '-', data['2016'].min())

plt.subplots(figsize=(13,8))



chart = plt.subplot(2, 2, 1)

withWorld  = np.array(data['2016'])

withoutWorld = np.delete(withWorld, 210, axis=0)

plt.plot(withoutWorld)



plt.ylabel('Population', fontsize=12)

plt.xlabel('Population Countries Distribution', fontsize=12)

plt.annotate("Circa 200 countries \n Distribution of countries \n smaller than 100 Million",

             xy=(0.45,0.95), xycoords='axes fraction', fontsize=10)



#Between 10 Millions and 100 Millions

result2 = data[np.logical_and(data['2016']>10000000, data['2016']<100000000)]

result2 = (result2['2016'])

#Between 1 Millions and 10 Millions

result3 = data[np.logical_and(data['2016']>1000000, data['2016']<10000000)]

result3 = (result3['2016'])

#Less than 1 Millions

result4 = data['2016']<1000000

result4= (data[result4]['2016'])



chart2 = plt.subplot(2, 2, 2)

result2.hist()

plt.setp(chart2, xticks = range(10000000,100000000,10000000), yticks=range(0,35,5))

plt.axvline(result2.mean(),linestyle='dashed',color='blue')

plt.ylabel('Number of Countries', fontsize=12)

plt.annotate("Countries between 10 and 100 Million \n Frequency distribution and median", 

             xy=(0.3,0.8), xycoords='axes fraction', fontsize=10)



chart3 = plt.subplot(2, 2, 3)

result3.hist()

plt.setp(chart3, xticks = range(1000000,10000000,1000000), yticks=range(0,35,5))

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

plt.axvline(result3.mean(),linestyle='dashed',color='blue')

plt.ylabel('Number of Countries', fontsize=12)

plt.annotate("Countries between 1 and 10 Million \n Frequency distribution and median", 

             xy=(0.45,0.8), xycoords='axes fraction', fontsize=10)



chart4 = plt.subplot(2, 2, 4)

result4.hist()

plt.annotate("Countries smaller than 1 Million \n Frequency distribution and median", 

             xy=(0.3,0.8), xycoords='axes fraction', fontsize=10)

plt.ylabel('Number of Countries', fontsize=12)

plt.setp(chart4, xticks = range(100000,1000000,100000), yticks=range(0,35,5))

plt.axvline(result4.mean(),linestyle='dashed',color='blue')

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))





plt.subplots(figsize=(13,4))

population=data['2016'].sort_values(ascending=False)[1:11].to_frame()

sns.barplot(data['Country'].values[population.index[0:11]], population['2016'], palette='inferno')

plt.title('Top 10 Countries by population')

plt.xlabel('')

plt.show()
population=data['2015'].sort_values(ascending=False)[1:6].to_frame()

worldPopulation = data['2015'].max()

sizes = (population['2015']/worldPopulation).iloc[::-1]

labels = data['Country'].values[population.index[0:6]][::-1]

explode = (0, 0, 0, 0, 0)

fig1, ax1 = plt.subplots(figsize=(13,4))

ax1.pie(sizes, radius = 1.1, explode=explode, labels=labels, labeldistance=1.1, 

autopct='%1.1f%%', shadow=False, startangle=-5)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()



plt.subplots(figsize=(13,8))

x = range(0, 216)

withoutWorld = withoutWorld.astype(float)

plt.scatter(x, withoutWorld, s = withoutWorld/1000000, c = withoutWorld)

plt.xlabel('Countries')

plt.ylabel('Population');
plt.subplots(figsize=(13,10))

population=data['2016'].sort_values(ascending=False)[1:11]

country = data['Country'].values[population.index[0:11]]

df = pd.DataFrame({'nb_people':population, 'group':country })

squarify.plot(sizes=df['nb_people'], label=df['group'], alpha=.8)

plt.axis('off')

plt.show()
plt.subplots(figsize=(13,8))

Mammals = ['Horse', 'Dogs', 'Cats', 'Goats', 'Pigs', 'Sheep', 'Cows', 'People']

MammalsPopulation = (60000000, 425000000, 625000000, 860000000, 1000000000, 1100000000, 1500000000, 7442135578)

Mammals = Mammals[::-1]

MammalsPopulation = MammalsPopulation[::-1]

squarify.plot(sizes=MammalsPopulation, label= Mammals, alpha=.9)

plt.axis('off')

plt.title('Top Mammals by population')

plt.show()    
plt.subplots(figsize=(13,8))

Mammals = ['Horse', 'Dogs', 'Cats', 'Goats', 'Pigs', 'Sheep', 'Cows', 'People', 'Chickens']

MammalsPopulation = (60000000, 425000000, 625000000, 860000000, 1000000000, 1100000000, 1500000000, 7330000000, 20000000000)

Mammals = Mammals[::-1]

MammalsPopulation = MammalsPopulation[::-1]

squarify.plot(sizes=MammalsPopulation, label= Mammals, alpha=.8)

plt.axis('off')

plt.title('Top Mammals by population')

plt.show()
plt.subplots(figsize=(13,8))

Mixed = ['Horse', 'Italy', 'Dogs', 'USA', 'Cats', 'EU', 'Sheep', 'India', 'Cows', 'China']

MixedPopulation = (60000000, (data[data["Country"]=="Italy"]["2016"].iloc[0]), 425000000, 

(data[data["Country"]=="USA"]["2016"].iloc[0]), 625000000, 

(data[data["Country"]=="EU"]["2016"].iloc[0]), 1100000000, 

(data[data["Country"]=="India"]["2016"].iloc[0]), 1500000000, 

(data[data["Country"]=="China"]["2016"].iloc[0]))



Mixed = Mixed[::-1]

MixedPopulation = MixedPopulation[::-1]

squarify.plot(sizes=MixedPopulation, label= Mixed, alpha=.9)

plt.axis('off')

plt.title('Mixed by population')

plt.show()
plt.subplots(figsize=(13,8))

population=data['2016'].to_frame()

indexesCountry = population.index[population['2016'] > 0]

numpyCountry = []

numpyGrowth = []

numpyMixed  = []

for index in indexesCountry:

     country = data['Country'].values[index]

     country1960 = (data[data["Country"]==country])["1960"].iloc[0]

     country2016 = (data[data["Country"]==country])["2016"].iloc[0]

     growthPercent = ((country2016-country1960)/country1960) * 100

     numpyCountry = np.append(numpyCountry, country)

     numpyGrowth = np.append(numpyGrowth, growthPercent)

     numpyMixed = np.vstack((numpyCountry, numpyGrowth)).T



x = range(0, len(numpyGrowth))

plt.scatter(x, numpyGrowth)

plt.ylabel('Population growth, %', labelpad=-2, fontsize=12)

plt.xlabel('Countries', fontsize=12)

plt.annotate("Circa 200 countries ", xy=(0.80,0.95), xycoords='axes fraction', fontsize=12)

plt.annotate("UAE", xy=(0.08, 0.9), xycoords='axes fraction', fontsize=10)

plt.annotate("Qatar", xy=(0.75, 0.55), xycoords='axes fraction', fontsize=10);
index = np.argsort(numpyMixed[:,1].astype(float))

countriesPopulationGrowthIncreasing = data['Country'].values[index]

countriesPopulationGrowthIncreasing = countriesPopulationGrowthIncreasing[0:212]



#correct here later

z = numpyMixed[numpyMixed[:,1].astype(float).argsort()]

x_sort = z[:,1].astype(float)

x_sort = x_sort[0:212]

sns.set(rc={'figure.figsize':(13,8)})

sns.barplot(countriesPopulationGrowthIncreasing, x_sort, log = False)

plt.title('Population Growth in Percents by Country')

plt.ylabel('Population Growth, %', fontsize=12)

#plt.axes().get_xaxis().set_ticks([])



#Keeps every n-th label

n = 26

[l.set_visible(False) for (i,l) in enumerate(plt.axes().xaxis.get_ticklabels()) if i % n != 0]

plt.show()





plt.subplots(figsize=(13,10))



#Less than 2000 percents

numpyGrowthRange1 = []

for element in numpyGrowth:

    if element < 2000:

        numpyGrowthRange1 = np.append(numpyGrowthRange1, element)

#print(numpyGrowthRange1)

#numpyGrowthRange1.shape



x = range(0, len(numpyGrowthRange1))

chart1 = plt.subplot(2, 2, 1)



plt.ylabel('Population Growth, %', fontsize=12)

plt.xlabel('Number of Countries', fontsize=12)

plt.annotate("All countries without \n UAE and Qatar \n World growth is 145% \n (horizontal line)", 

             xy=(0.7,0.8), xycoords='axes fraction', fontsize=10)

plt.scatter(x, numpyGrowthRange1, alpha=0.9)

plt.axhline(145,linestyle='dashed',color='blue')



#Between 100 and 1000 percents

numpyGrowthRange2 = []

for element in numpyGrowth:

    if element < 1000 and element > 100:

        numpyGrowthRange2 = np.append(numpyGrowthRange2, element)

x = range(0, len(numpyGrowthRange2))

chart2 = plt.subplot(2, 2, 2)



plt.ylabel('Population Growth, %', fontsize=12)

plt.xlabel('Number of Countries', fontsize=12)

plt.annotate("Circa 140 countries grown \n between 100 and 1000 percents", xy=(0.6,0.9), 

             xycoords='axes fraction', fontsize=10)

plt.scatter(x, numpyGrowthRange2, alpha=0.9)

#plt.axhline(numpyGrowthRange2.mean(),linestyle='dashed',color='blue')



#Between 0 and 100 percents

numpyGrowthRange3 = []

for element in numpyGrowth:

    if element < 100 and element > 0:

        numpyGrowthRange3 = np.append(numpyGrowthRange3, element)

x = range(0, len(numpyGrowthRange3))

chart3 = plt.subplot(2, 2, 3)



plt.ylabel('Population Growth, %', fontsize=12)

plt.xlabel('Number of Countries', fontsize=12)

plt.annotate("Circa 70 countries grown \n between 0 and 100 percents", xy=(0.6,0.95), 

             xycoords='axes fraction', fontsize=10)

plt.scatter(x, numpyGrowthRange3, alpha=0.9)



#Population decline

numpySelected = numpyMixed[numpyMixed[:,1].astype(float) < 0.0]

x4 = range(1, len(numpySelected)+1)

y4 = np.around(numpySelected[:,1].astype(float), decimals = 2)

chart4 = plt.subplot(2, 2, 4)

plt.annotate("3 countries lost their population", xy=(0.05,0.95), xycoords='axes fraction', fontsize=10)

plt.scatter(x4, y4, alpha=0.9)

countries = numpySelected[:,0]



for i, country in enumerate(countries):

    plt.annotate(country, (x4[i], y4[i]))



plt.ylabel('Population Decline, %', fontsize=12)

plt.xlabel('Countries', fontsize=12)

plt.xticks([])

chart4.set_ylim(ymin=-10)

chart4.set_ylim(ymax=0);
plt.subplots(figsize=(13,8))

population=data['2016'].sort_values(ascending=False)[1:11].to_frame()

indexesCountry = population.index[population['2016'] > 0]

numpyCountry = []

numpyGrowth = []

numpyMixed  = []

for index in indexesCountry:

     country = data['Country'].values[index]

     #print(country)

     country1960 = (data[data["Country"]==country])["1960"].iloc[0]

     country2016 = (data[data["Country"]==country])["2016"].iloc[0]

     #print('1960:', country1960)

     #print('2016', country2016)

     growthPercent = ((country2016-country1960)/country1960) * 100

     #print("Growth in Percents", growthPercent)

     numpyCountry = np.append(numpyCountry, country)

     numpyGrowth = np.append(numpyGrowth, growthPercent)



sns.barplot(numpyCountry, numpyGrowth, palette='inferno')

plt.title('Population Growth in Percents by Country')

plt.ylabel('Population Growth, %', fontsize=12)

plt.show();
stringChina = (data[data["Country"]=="China"])

growthChina = stringChina.loc[:, '1960':'2016'].T

stringIndia = (data[data["Country"]=="India"])

growthIndia = stringIndia.loc[:, '1960':'2016'].T

stringEU = (data[data["Country"]=="EU"])

growthEU = stringEU.loc[:, '1960':'2016'].T

ax = growthChina.plot(figsize=(13,8), linewidth=5)

growthIndia.plot(ax=ax, linewidth=5)

growthEU.plot(ax=ax, linewidth=5)

ax.legend('')

legend = plt.legend(["China", "India", "EU"], loc=2)

ax.set_ylim(ymin=0)

plt.ylabel('Population growth of most populated countries')

plt.xlabel('Years')

plt.show()
stringUSA = (data[data["Country"]=="USA"])

growthUSA = stringUSA.loc[:, '1960':'2016'].T

stringIndonesia = (data[data["Country"]=="Indonesia"])

growthIndonesia = stringIndonesia.loc[:, '1960':'2016'].T

stringBrazil = (data[data["Country"]=="Brazil"])

growthBrazil = stringBrazil.loc[:, '1960':'2016'].T

stringPakistan = (data[data["Country"]=="Pakistan"])

growthPakistan = stringPakistan.loc[:, '1960':'2016'].T

stringNigeria = (data[data["Country"]=="Nigeria"])

growthNigeria = stringNigeria.loc[:, '1960':'2016'].T

stringBangladesh = (data[data["Country"]=="Bangladesh"])

growthBangladesh = stringBangladesh.loc[:, '1960':'2016'].T

stringRussia = (data[data["Country"]=="Russia"])

growthRussia = stringRussia.loc[:, '1960':'2016'].T



ax = growthUSA.plot(figsize=(13,8), linewidth=4)

growthIndonesia.plot(ax=ax, linewidth=4)

growthBrazil.plot(ax=ax, linewidth=4)

growthPakistan.plot(ax=ax, linewidth=4)

growthNigeria.plot(ax=ax, linewidth=4)

growthBangladesh.plot(ax=ax, linewidth=4)

growthRussia.plot(ax=ax, linewidth=4)

ax.legend('')

legend = plt.legend(["USA", "Indonesia", "Brazil", "Pakistan", "Nigeria", "Bangladesh", "Russia"], loc=2)

ax.set_ylim(ymin=0)

#plt.show()

plt.ylabel('Population growth of most populated countries')

plt.xlabel('Years');
population=data[data['2016']<2000000000]



metricscale1=[[0, 'rgb(102,194,165)'], [0.05, 'rgb(102,194,165)'],

              [0.15, 'rgb(171,221,164)'], [0.2, 'rgb(230,245,152)'],

              [0.25, 'rgb(255,255,191)'], [0.35, 'rgb(254,224,139)'],

              [0.45, 'rgb(253,174,97)'], [0.55, 'rgb(213,62,79)'], [1.0, 'rgb(158,1,66)']]

data = [ dict(

        type = 'choropleth',

        autocolorscale = False,

        colorscale = metricscale1,

        showscale = True,

        locations = population['Country'].values,

        z = population['2016'].values,

        locationmode = 'country names',

        text = population['Country'].values,

        marker = dict(

            line = dict(color = 'rgb(250,250,225)', width = 0.5)),

            colorbar = dict(autotick = True, tickprefix = '',

            title = 'Population')

            )

       ]



layout = dict(

    title = 'World Population Map, 2016',

    geo = dict(

        showframe = True,

        showocean = True,

        oceancolor = 'rgb(28,107,160)',

        projection = dict(

        type = 'orthographic',

            rotation = dict(

                    lon = 60,

                    lat = 10),

        ),

        lonaxis =  dict(

                showgrid = False,

                gridcolor = 'rgb(102, 102, 102)'

            ),

        lataxis = dict(

                showgrid = False,

                gridcolor = 'rgb(102, 102, 102)'

                )

            ),

        )

fig = dict(data=data, layout=layout)

py.iplot(fig, validate=False, filename='worldmap2016')   

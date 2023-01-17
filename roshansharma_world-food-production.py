# for basic operations

import numpy as np

import pandas as pd



# for visualizations

import matplotlib.pyplot as plt

import seaborn as sns



# for advanced visualizations

import folium

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



# for providing path

import os

print(os.listdir('../input/'))


# reading the data

data = pd.read_csv('../input/FAO.csv', encoding = "ISO-8859-1")



# ENODING ISO-8859-1 is a single byte encoding which can represent the first 256 unicode characters

# Both UTF-8 and ISO-8859-1 encode the ASCII Characters the same.



# checking the shape of the data

print(data.shape)
data.head()
# adding a total production column



data['total'] = (data['Y1961'] + data['Y1962'] + data['Y1963'] + data['Y1964'] + data['Y1965'] + data['Y1966'] + 

    data['Y1967'] + data['Y1968'] + data['Y1969'] + data['Y1970'] + data['Y1971'] + data['Y1972'] + data['Y1973'] +

    data['Y1974'] + data['Y1975'] + data['Y1976'] + data['Y1977'] + data['Y1978'] + data['Y1979'] + data['Y1980'] + 

    data['Y1981'] + data['Y1982'] + data['Y1983'] + data['Y1984'] + data['Y1985'] + data['Y1986'] + data['Y1987'] + 

    data['Y1988'] + data['Y1989'] + data['Y1990'] + data['Y1991'] + data['Y1992'] + data['Y1993'] + data['Y1994'] + 

    data['Y1995'] + data['Y1996'] + data['Y1997'] + data['Y1998'] + data['Y1999'] + data['Y2000'] + data['Y2001'] + 

    data['Y2001'] + data['Y2002'] + data['Y2003'] + data['Y2004'] + data['Y2005'] + data['Y2006'] + data['Y2007'] + 

    data['Y2008'] + data['Y2009'] + data['Y2010'] + data['Y2011'] + data['Y2012'] + data['Y2013'] )
data.describe()
df = data['Area'].value_counts().sort_index().index

df2 = data.groupby('Area')['total'].agg('mean')



trace = go.Choropleth(

    locationmode = 'country names',

    locations = df,

    text = df,

    colorscale = 'Picnic',

    z = df2.values

)

df3 = [trace]

layout = go.Layout(

    title = 'Mean Production in Differet Parts of World')



fig = go.Figure(data = df3, layout = layout)

iplot(fig)
df = data['Area'].value_counts().sort_index().index

df2 = data.groupby('Area')['Y1961'].agg('mean')



trace = go.Choropleth(

    locationmode = 'country names',

    locations = df,

    text = df,

    colorscale = 'Rainbow',

    z = df2.values

)

df3 = [trace]

layout = go.Layout(

    title = 'Mean Production in 1961 in Differet Parts of World')



fig = go.Figure(data = df3, layout = layout)

iplot(fig)
df = data['Area'].value_counts().sort_index().index

df2 = data.groupby('Area')['Y2013'].agg('mean')



trace = go.Choropleth(

    locationmode = 'country names',

    locations = df,

    text = df,

    colorscale = 'Hot',

    z = df2.values

)

df3 = [trace]

layout = go.Layout(

    title = 'Mean Production in 2013 in Differet Parts of World')



fig = go.Figure(data = df3, layout = layout)

iplot(fig)
# delete the total column



data = data.drop(['total'], axis = 1)

color = plt.cm.Wistia(np.linspace(0, 1, 40))

plt.style.use('_classic_test')



data['Area'].value_counts().sort_values(ascending = False).head(40).plot.bar(figsize = (20, 10), color = color)

plt.title('Number of Different Items produced by Different Countries in the World', fontsize = 20)

plt.xlabel('Name of the Countries', fontsize = 10)

plt.show()
# Top Products around the globe



# setting the style to be ggplot

plt.style.use("dark_background")



items = pd.DataFrame(data.groupby("Item")["Element"].agg("count").sort_values(ascending=False))[:100]



# plotting

plt.rcParams['figure.figsize'] = (15, 20)

#plt.gcf().subplots_adjust(left = .3)

sns.barplot(x = items.Element, y = items.index, data = items, palette = 'Reds')

plt.gca().set_title("Top 100 items produced around globe", fontsize = 30)

plt.show()
# setting the size of the plot

plt.rcParams['figure.figsize'] = (20, 20)





# looking at India's Growth

india_production = pd.DataFrame(data[data['Area'] == 'India'].loc[:, "Y2003": "Y2013"].agg("sum", axis = 0))



india_production.columns = ['Production']

plt.subplot(231)

sns.barplot(x = india_production.index, y = india_production.Production, data = india_production, palette = 'PuBu')

plt.gca().set_title("India's Growth")



# looking at china's growth

china_production = pd.DataFrame(data[data['Area'] == 'China, mainland'].loc[:, "Y2003":"Y2013"].agg("sum", axis = 0))



china_production.columns = ['Production']

plt.subplot(232)

sns.barplot(x = china_production.index, y = india_production.Production, data = china_production, palette = 'RdPu')

plt.gca().set_title("China's Growth")



#looking at usa's growth

usa_production = pd.DataFrame(data[data['Area'] == 'United States of America'].loc[:,"Y2003":"Y2013"].agg("sum", axis = 0))



usa_production.columns = ['Production']

plt.subplot(233)

sns.barplot(x = usa_production.index, y = usa_production.Production, data = usa_production, palette = 'Blues')

plt.gca().set_title("USA's Growth")



#looking at brazil's growth

brazil_production = pd.DataFrame(data[data['Area'] == 'Brazil'].loc[:,"Y2003":"Y2013"].agg("sum", axis = 0))



brazil_production.columns = ['Production']

plt.subplot(234)

sns.barplot(x = brazil_production.index, y = brazil_production.Production, data = brazil_production, palette = 'Purples')

plt.gca().set_title("Brazil's Growth")





#looking at mexico's growth

mexico_production = pd.DataFrame(data[data['Area'] == 'Mexico'].loc[:,"Y2003":"Y2013"].agg("sum", axis = 0))



mexico_production.columns = ['Production']

plt.subplot(235)

sns.barplot(x = mexico_production.index, y = mexico_production.Production, data = mexico_production, palette = 'ocean')

plt.gca().set_title("Mexico's Growth")



#looking at russia's growth

russia_production = pd.DataFrame(data[data['Area'] == 'Russian Federation'].loc[:,"Y2003":"Y2013"].agg("sum", axis = 0))



russia_production.columns = ['Production']

plt.subplot(236)

sns.barplot(x = russia_production.index, y = russia_production.Production, data = russia_production, palette = 'spring')

plt.gca().set_title("Russia's Growth")



plt.suptitle('Top 6 Countries Growth from 2003 to 2013', fontsize = 30)

plt.show()
labels = ['Feed', 'Food']

size = data['Element'].value_counts()

colors = ['cyan', 'magenta']

explode = [0.1, 0.1]



plt.rcParams['figure.figsize'] = (10, 10)

plt.pie(size, labels = labels, colors = colors, explode = explode, shadow = True)

plt.axis('off')

plt.title('A Pie Chart Representing Types of Elements', fontsize = 20)

plt.legend()

plt.show()
# plotting for the Annual Production of crops by every country



countries = list(data['Area'].unique())

years = list(data.iloc[:, 10:].columns)



plt.style.use('seaborn')    

plt.figure(figsize = (20, 20))

for i in countries:

    production = []

    for j in years:

        production.append(data[j][data['Area'] == i].sum())

    plt.plot(production, label = i)

    

plt.xticks(np.arange(53), tuple(years), rotation = 90)

plt.title('Country wise Annual Production')

plt.legend()

plt.legend(bbox_to_anchor = (0., 1, 1.5,  1.5), loc = 3, ncol = 12)

plt.savefig('p.png')

plt.show()
# creating a new data containing information about countries and productions only



new_data_dict = {}

for i in countries:

    production = []

    for j in years:

        production.append(data[j][data['Area'] == i].sum())

    new_data_dict[i] = production

new_data = pd.DataFrame(new_data_dict)



new_data.head()
new_data['Year'] = np.linspace(1961, 2013, num = 53).astype('int')



# checking the shape of the new data

new_data.shape


#heatmap



plt.rcParams['figure.figsize'] = (15, 15)

plt.style.use('fivethirtyeight')

 

sns.heatmap(new_data, cmap = 'PuBu')

plt.title('Heatmap for Production', fontsize = 20)

plt.yticks()

plt.show()
plt.rcParams['figure.figsize'] = (15, 8)

plt.style.use('dark_background')



sns.lineplot(new_data['Year'], new_data['United States of America'], color = 'yellow')

plt.title('Time Series Analysis for USA', fontsize = 30)

plt.grid()

plt.show()
plt.rcParams['figure.figsize'] = (15, 8)

plt.style.use('dark_background')



sns.lineplot(new_data['Year'], new_data['India'], color = 'yellow')

plt.title('Time Series Analysis for India', fontsize = 30)

plt.grid()

plt.show()
plt.rcParams['figure.figsize'] = (15, 8)

plt.style.use('dark_background')



sns.lineplot(new_data['Year'], new_data['China, mainland'], color = 'yellow')

plt.title('Time Series Analysis for China', fontsize = 30)

plt.grid()

plt.show()
plt.rcParams['figure.figsize'] = (15, 8)

plt.style.use('dark_background')



sns.lineplot(new_data['Year'], new_data['Russian Federation'], color = 'yellow')

plt.title('Time Series Analysis for Russia', fontsize = 30)

plt.grid()

plt.show()
plt.rcParams['figure.figsize'] = (15, 8)

plt.style.use('dark_background')



sns.lineplot(new_data['Year'], new_data['Iceland'], color = 'yellow')

plt.title('Time Series Analysis for Iceland', fontsize = 30)

plt.grid()

plt.show()
plt.rcParams['figure.figsize'] = (15, 8)

plt.style.use('dark_background')



sns.lineplot(new_data['Year'], new_data['Brazil'], color = 'yellow')

plt.title('Time Series Analysis for Brazil', fontsize = 30)

plt.grid()

plt.show()
















































































































































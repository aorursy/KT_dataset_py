# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
happiness2015 = pd.read_csv("../input/2015.csv", index_col=0)
print(happiness2015.columns)
happiness2016 = pd.read_csv("../input/2016.csv", index_col=0)
print(happiness2016.columns)
happiness2017 = pd.read_csv("../input/2017.csv", index_col=0)
print(happiness2017.columns)
# Check for missing values
columns_missing_values_2015 = [col for col in happiness2015.columns 
                                 if happiness2015[col].isnull().any()]
columns_missing_values_2016 = [col for col in happiness2016.columns 
                                 if happiness2016[col].isnull().any()]
columns_missing_values_2017 = [col for col in happiness2017.columns 
                                 if happiness2017[col].isnull().any()]
print(columns_missing_values_2015)
print(columns_missing_values_2016)
print(columns_missing_values_2017)
import seaborn as sns
import matplotlib.pyplot as plt


# 2015 HeatMap
heatmap2015 = sns.heatmap(
    happiness2015.loc[:, ['Happiness Score',
       'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)',
       'Freedom', 'Trust (Government Corruption)', 'Generosity',
       'Dystopia Residual']].corr(method='pearson'),
    cmap="YlGnBu",
    annot=True
)

plt.title("2015 HeatMap")
plt.show(heatmap2015)

# 2016 HeatMap
heatmap2016 = sns.heatmap(
    happiness2016.loc[:, [ 'Happiness Score',
       'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)',
       'Freedom', 'Trust (Government Corruption)', 'Generosity',
       'Dystopia Residual']].corr(method='pearson'),
    cmap="YlGnBu",
    annot=True
)

plt.title("2016 HeatMap")
plt.show(heatmap2016)


# 2017 HeatMap
heatmap2017 = sns.heatmap(
    happiness2017.loc[:, ['Happiness.Score',
       'Economy..GDP.per.Capita.', 'Family', 'Health..Life.Expectancy.',
       'Freedom', 'Trust..Government.Corruption.', 'Generosity',
       'Dystopia.Residual']].corr(method='pearson'),
    cmap="YlGnBu",
    annot=True
)


plt.title("2017 HeatMap")
plt.show(heatmap2017)

# 2015 World Map
import pycountry
import plotly.plotly as py  # plot package
import plotly
from plotly.offline import iplot, init_notebook_mode  #iplot is for plotting into a jupyter
init_notebook_mode()

data2015 = pd.read_csv('../input/2015.csv')
print(data2015.columns)
data2015 = data2015[['Country', 'Happiness Score', 'Generosity', 'Freedom', 'Trust (Government Corruption)', 'Family']]
countries= data2015['Country'].values
print(countries)
def lookup(countries):
    result = []
    for i in range(len(countries)):
        try:
            result.append(pycountry.countries.get(name=countries[i]).alpha_3)
        except KeyError:
            try:
                result.append(pycountry.countries.get(official_name=countries[i]).alpha_3)
            except KeyError:
                result.append('undefined')
    return result

iso3codes=lookup(countries)
print(iso3codes)

data2015['Codes']=iso3codes
data2015=data2015[~data2015.Codes.isin(['undefined'])]
print(data2015.Codes)
mapdata2015Generosity = [ dict(
    type = 'choropleth',
    locations = data2015['Codes'],
    z = data2015['Generosity'],
    text = data2015['Country'],
   colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"], \
                 [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
   autocolorscale = False,
   reversescale = True,
    marker = dict(
       line = dict (
           color = 'rgb(180,180,180)',
           width = 0.5
       ) ),
    colorbar = dict(
       autotick = False,
       title = 'Generosity'),
) ]

layoutGenerosity = dict(
    title = 'Generosity 2015',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)
fig = dict( data=mapdata2015Generosity, layout=layoutGenerosity )
iplot(fig, validate=False)
mapdata2015Trust = [ dict(
    type = 'choropleth',
    locations = data2015['Codes'],
    z = data2015['Trust (Government Corruption)'],
    text = data2015['Country'],
   colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"], \
                 [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
   autocolorscale = False,
   reversescale = True,
    marker = dict(
       line = dict (
           color = 'rgb(180,180,180)',
           width = 0.5
       ) ),
    colorbar = dict(
       autotick = False,
       title = 'Trust (Government Corruption)'),
) ]

layoutTrust = dict(
    title = 'Trust 2015',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=mapdata2015Trust, layout=layoutTrust )
iplot(fig, validate=False)
mapdata2015Freedom = [ dict(
    type = 'choropleth',
    locations = data2015['Codes'],
    z = data2015['Freedom'],
    text = data2015['Country'],
   colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"], \
                 [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
   autocolorscale = False,
   reversescale = True,
    marker = dict(
       line = dict (
           color = 'rgb(180,180,180)',
           width = 0.5
       ) ),
    colorbar = dict(
       autotick = False,
       title = 'Freedom'),
) ]

layoutFreedom = dict(
    title = 'Freedom 2015',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=mapdata2015Freedom, layout=layoutFreedom )
iplot(fig, validate=False)
mapdata2015HappinessScore = [ dict(
    type = 'choropleth',
    locations = data2015['Codes'],
    z = data2015['Happiness Score'],
    text = data2015['Country'],
   colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"], \
                 [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
   autocolorscale = False,
   reversescale = True,
    marker = dict(
       line = dict (
           color = 'rgb(180,180,180)',
           width = 0.5
       ) ),
    colorbar = dict(
       autotick = False,
       title = 'Happiness Score'),
) ]

layoutHappinessScore = dict(
    title = 'Happiness Score 2015',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=mapdata2015HappinessScore, layout=layoutHappinessScore )
iplot(fig, validate=False)

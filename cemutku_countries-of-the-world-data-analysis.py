import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv),
import types

# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# word cloud library
from wordcloud import WordCloud

# matplotlib
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
countryData = pd.read_csv("../input/countries of the world.csv")
countryData.info()
countryData.head(10)
africaData = countryData[countryData["Region"].str.contains('AFRICA')]
africaData = africaData.sort_values(["GDP ($ per capita)"],ascending = False)
trace1 = go.Scatter(
                    x = africaData["GDP ($ per capita)"],
                    y = africaData["Birthrate"],
                    mode = "lines",
                    name = "Birthrate",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                    text = africaData.Country)

trace2 = go.Scatter(
                    x = africaData["GDP ($ per capita)"],
                    y = africaData["Deathrate"],
                    mode = "lines+markers",
                    name = "Deathrate",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text = africaData.Country)

data = [trace1, trace2]
layout = dict(title = 'Birth and Death Rates of African Countries', xaxis = dict(title='GDP ($ per capita)', ticklen = 5, zeroline = False))
fig = dict(data = data, layout = layout)
iplot(fig)
def formatFloat(value):    
    if isinstance(value, str):
        return float(value.replace(',', '.'))
    else:
        return float(value)

europeData = countryData[countryData["Region"].str.contains('EUROPE')]
europeData["Literacy (%)"].fillna(0, inplace = True)
europeData["Literacy (%)"] = [formatFloat(i) for i in europeData["Literacy (%)"]]
asiaData = countryData[countryData["Region"].str.contains('ASIA')]
asiaData["Literacy (%)"].fillna(0, inplace = True)
asiaData["Literacy (%)"] = [formatFloat(i) for i in asiaData["Literacy (%)"]]
latinAmericaData = countryData[countryData["Region"].str.contains('LATIN')]
latinAmericaData["Literacy (%)"].fillna(0, inplace = True)
latinAmericaData["Literacy (%)"] = [formatFloat(i) for i in latinAmericaData["Literacy (%)"]]
trace1 = go.Scatter(
                    x = europeData["GDP ($ per capita)"],
                    #x = europeData["Population"],
                    y = europeData["Literacy (%)"],
                    mode = "markers",
                    name = "Europe",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text = europeData.Country)

trace2 = go.Scatter(
                    x = asiaData["GDP ($ per capita)"],
                    #x = asiaData["Population"],
                    y = asiaData["Literacy (%)"],
                    mode = "markers",
                    name = "Asia",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text = asiaData.Country)

trace3 = go.Scatter(
                    x = latinAmericaData["GDP ($ per capita)"],
                    #x = latinAmericaData["Population"],
                    y = latinAmericaData["Literacy (%)"],
                    mode = "markers",
                    name = "Latin America",
                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),
                    text = latinAmericaData.Country)

data = [trace1, trace2, trace3]
layout = dict(title = 'Europe - Asia - Latin America Literacy Comparison (%)',
              xaxis = dict(title = 'GDP ($ per capita)', ticklen = 5, zeroline = False),
              yaxis = dict(title = 'Literacy (%)', ticklen = 5, zeroline = False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
areaData = countryData.sort_values(["Area (sq. mi.)"],ascending = False).iloc[:5,:]
areaData["Pop. Density (per sq. mi.)"] = [formatFloat(i) for i in areaData["Pop. Density (per sq. mi.)"]]
areaData = areaData.sort_values(["Population"], ascending = False)

areaData
trace1 = go.Bar(
                x = areaData.Country,
                y = areaData.Birthrate,
                name = "Birthrate",
                marker = dict(color = 'rgba(255, 174, 255, 0.5)', line = dict(color = 'rgb(0, 0, 0)', width = 1.5)),
                text = areaData.Country)

trace2 = go.Bar(
                x = areaData.Country,
                y = areaData.Deathrate,
                name = "Deathrate",
                marker = dict(color = 'rgba(0, 0, 128, 0.5)', line = dict(color = 'rgb(0, 0, 0)', width = 1.5)),
                text = areaData.Country)

data = [trace1, trace2]
layout = go.Layout(barmode = "group", 
                   title = 'Birthrate and Deathrate of Top 5 Most Crowded Countries',
                   xaxis = dict(title = 'Country', ticklen = 5, zeroline = False),
                   yaxis = dict(title = 'Rates', ticklen = 5, zeroline = False))

fig = go.Figure(data = data, layout = layout)
iplot(fig)
x = areaData.Country

trace1 = {
  'x': x,
  'y': areaData.Birthrate,
  'name': 'Birthrate',
  'type': 'bar'
};
trace2 = {
  'x': x,
  'y': areaData.Deathrate,
  'name': 'Deathrate',
  'type': 'bar'
};
data = [trace1, trace2];
layout = {
  'xaxis': {'title': 'Top 5 Countries'},
  'barmode': 'relative',
  'title': 'Birthrate and Deathrate of Top 5 Most Crowded Countries'
};
fig = go.Figure(data = data, layout = layout)
iplot(fig)
# data preparation
capitalData = countryData.sort_values(["GDP ($ per capita)"],ascending = False)
capitalData = capitalData.iloc[:5,:]
labels = capitalData.Country
capitalData.head()
# figure
fig = {
  "data": [
    {
      "values": capitalData["GDP ($ per capita)"],
      "labels": labels,
      "domain": {"x": [0, .5]},
      "name": "GDP ($ per capita)",
      "hoverinfo":"label + percent + name",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"Top 5 Countries With Highest GDP",
        "annotations": [
            {"font": {"size": 20},
              "showarrow": False,
              "text": "GDP ($ per capita)",
                "x": 0.20,
                "y": 1
            },
        ]
    }
}
iplot(fig)
europeData = europeData.sort_values(["Area (sq. mi.)"],ascending = False)
europeData["Pop. Density (per sq. mi.)"] = [formatFloat(i) for i in europeData["Pop. Density (per sq. mi.)"]]
europeData["Arable (%)"].fillna(0, inplace = True)
europeData["Arable (%)"] = [formatFloat(i) for i in europeData["Arable (%)"]]
europeData["Agriculture"].fillna(0, inplace = True)
europeData["Agriculture"] = [(formatFloat(i)*100) for i in europeData["Agriculture"]]
data = [
    {
        'y': europeData.Population,
        'x': europeData["Area (sq. mi.)"],
        'mode': 'markers',
        'marker': {
            'color': europeData["Agriculture"],
            'size': europeData["Arable (%)"],
            'showscale': True
        },
        "text" :  europeData.Country    
    }
]
iplot(data)
# data preparation
westernEuropeData = countryData[countryData["Region"].str.contains('WESTERN EUROPE')]
easternEuropeData = countryData[countryData["Region"].str.contains('EASTERN EUROPE')]
trace1 = go.Histogram(
    x = westernEuropeData["GDP ($ per capita)"],
    opacity = 0.75,
    name = "WESTERN EUROPE",
    marker = dict(color = 'rgba(171, 50, 96, 0.6)'),
    text = westernEuropeData.Country)

trace2 = go.Histogram(
    x = easternEuropeData["GDP ($ per capita)"],
    opacity = 0.75,
    name = "EASTERN EUROPE",
    marker = dict(color = 'rgba(12, 50, 196, 0.6)'),
    text = easternEuropeData.Country)

data = [trace1, trace2]
layout = go.Layout(barmode = 'overlay',
                   title = '-----',
                   xaxis=  dict(title = 'students-staff ratio'),
                   yaxis = dict(title = 'Count'),
)
fig = go.Figure(data = data, layout = layout)
iplot(fig)
regions = countryData.Region
plt.subplots(figsize = (8, 8))
wordcloud = WordCloud(
                          background_color = 'white',
                          width = 512,
                          height = 384
                         ).generate("".join(regions))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')
plt.show()
trace1 = go.Box(
    y = europeData["Arable (%)"],
    name = 'Population',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)

trace2 = go.Box(
    y = europeData["Agriculture"],
    name = 'Area (sq. mi.)',
    marker = dict(
        color = 'rgb(12, 128, 128)',
    )
)

data = [trace1, trace2]
iplot(data)
# import figure factory
import plotly.figure_factory as ff

# prepare data
scatterData = europeData.loc[:,["Arable (%)", "Agriculture", "Area (sq. mi.)"]]
scatterData["index"] = np.arange(1, len(scatterData) + 1)

# scatter matrix
fig = ff.create_scatterplotmatrix(scatterData, 
                                  diag = 'box', 
                                  index = 'index', 
                                  colormap = 'Portland',
                                  colormap_type = 'cat',
                                  height = 700, 
                                  width = 700)
iplot(fig)
# data preparation
europeData = europeData.sort_values(["GDP ($ per capita)"], ascending = False)

# first line plot
trace1 = go.Scatter(
    x = europeData["GDP ($ per capita)"],
    y = europeData.Agriculture,
    name = "Agriculture",
    text = europeData.Country,
    marker = dict(color = 'rgba(16, 112, 2, 0.8)')
)

# second line plot
trace2 = go.Scatter(
    x = europeData["GDP ($ per capita)"],
    y = europeData["Arable (%)"],
    xaxis = 'x2',
    yaxis = 'y2',
    name = "Arable(%)",
    text = europeData.Country,
    marker = dict(color = 'rgba(160, 112, 20, 0.8)')
)

# combine trace1 and trace2
data = [trace1, trace2]
layout = go.Layout(
    xaxis2 = dict(domain = [0.6, 0.95], anchor = 'y2'),
    yaxis2 = dict(domain = [0.6, 0.95], anchor = 'x2'),
    title = 'Agriculture and Arable(%) vs GDP of Europen Countries'
)

fig = go.Figure(data = data, layout = layout)
iplot(fig)
europeData.head()
# create trace 1 that is 3d scatter
trace1 = go.Scatter3d(
    x = europeData["GDP ($ per capita)"],
    y = europeData["Arable (%)"],
    z = europeData["Agriculture"],
    mode = 'markers',
    text = europeData.Country,
    marker = dict(
        size = 10,
        color = europeData["GDP ($ per capita)"],
        colorscale = 'Bluered'
    )
)

data = [trace1]
layout = go.Layout(
    margin = dict(
        l = 0,
        r = 0,
        b = 0,
        t = 0  
    )    
)

fig = go.Figure(data = data, layout = layout)
iplot(fig)
trace1 = go.Scatter(
    x = europeData["GDP ($ per capita)"],
    y = europeData["Arable (%)"],
    text = europeData.Country,
    name = "Arable (%)"
)

trace2 = go.Scatter(
    x = europeData["GDP ($ per capita)"],
    y = europeData["Agriculture"],
    xaxis = 'x2',
    yaxis = 'y2',
    text = europeData.Country,
    name = "Agriculture"
)

trace3 = go.Scatter(
    x = europeData["GDP ($ per capita)"],
    y = europeData["Area (sq. mi.)"],
    xaxis = 'x3',
    yaxis = 'y3',
    text = europeData.Country,
    name = "Area (sq. mi.)"
)

trace4 = go.Scatter(
    x = europeData["GDP ($ per capita)"],
    y = europeData["Population"],
    xaxis = 'x4',
    yaxis = 'y4',
    text = europeData.Country,
    name = "Population"
)

data = [trace1, trace2, trace3, trace4]
layout = go.Layout(
    xaxis = dict(
        domain = [0, 0.45]
    ),
    yaxis= dict(
        domain = [0, 0.45]
    ),
    xaxis2 = dict(
        domain = [0.55, 1]
    ),
    yaxis2 = dict(
        domain = [0, 0.45],
        anchor = 'x2'
    ),
    xaxis3 = dict(
        domain = [0, 0.45],
        anchor = 'y3'
    ),
    yaxis3 = dict(
        domain = [0.55, 1]
    ),
    xaxis4 = dict(
        domain = [0.55, 1],
        anchor = 'y4'
    ),
    yaxis4 = dict(
        domain = [0.55, 1],
        anchor = 'x4'
    ),
    title = 'Arable (%), Agriculture, Area (sq. mi.) and Population vs GDP ($ per capita) of Europen Countries'
)

fig = go.Figure(data = data, layout = layout)
iplot(fig)
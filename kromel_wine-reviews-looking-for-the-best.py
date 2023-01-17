# Import libraries
import numpy as np
import pandas as pd

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
wines = pd.read_csv("../input/winemag-data-130k-v2.csv")
ax = sns.heatmap(wines.isnull().T, xticklabels=False, cbar=False)
ax.set_title('Missing data')
plt.show()
wines = wines.drop(['designation', 'region_2', 'taster_name', 'taster_twitter_handle', 'Unnamed: 0'], axis=1)
wines = wines[wines['price'].notnull()]
price_hist = go.Histogram(
                x=wines[wines['price'] < 200]['price'], 
                nbinsx = 40
            )

data = [price_hist]

layout = go.Layout(
    title='Price distribution',
    autosize=False,
    width=800,
    height=350
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
plt.subplots(figsize=(12,2))
ax = sns.regplot(x="price", y="points", data=wines, scatter_kws={'alpha':0.4, 's': 13}, fit_reg=False)
ax.set_ylim(79, 101)
ax.set_xlim(0, 1000)
ax.set_yticks([80, 85, 90, 95, 100])
ax.set_title('Impact of price on taste(points)')
plt.show()
# creating price_range feature
wines['price_range'] = 'high'
wines.loc[wines['price'] < 49, 'price_range'] = 'medium'
wines.loc[wines['price'] < 14, 'price_range'] = 'low'
price_range_cats = ['low', 'medium', 'high']
price_range_boxes = []

for cat in price_range_cats:
    obj = go.Box(
        y=wines[wines['price_range'] == cat]['points'],
        name=cat
    )
    price_range_boxes.append(obj)

layout = go.Layout(
    title='Points distribution in price categories',
    autosize=False,
    width=800,
    height=350
)

fig = go.Figure(data=price_range_boxes, layout=layout)
py.iplot(fig)
wines['taste/price'] = wines['points']/wines['price']
plt.subplots(figsize=(12,2))
ax1 = sns.regplot(x="taste/price", y="points", data=wines[wines['price_range'] == 'high'], label="high",
                 scatter_kws={'alpha':0.5, 's': 3, 'color': 'green', 'marker': 'o'}, fit_reg=False)
ax2 = sns.regplot(x="taste/price", y="points", data=wines[wines['price_range'] == 'medium'], label="medium",
                 scatter_kws={'alpha':0.5, 's': 3, 'color': 'blue', 'marker': 'o'}, fit_reg=False)
ax3 = sns.regplot(x="taste/price", y="points", data=wines[wines['price_range'] == 'low'], label="low",
                 scatter_kws={'alpha':0.5, 's': 3, 'color': 'red', 'marker': 'o'}, fit_reg=False)

plt.xlim(0, 22)
plt.yticks([80, 85, 90, 95, 100])
plt.title('Relation between number of points and taste/price ratio')
plt.legend(markerscale=3)
plt.show()
wine_variety_15 = wines['variety'].value_counts()[:10]
price_variety_box = []

for variety in wine_variety_15.index:
    obj = go.Box(
        x=wines[(wines['variety'] == variety)]['price'],
        name=variety,
        orientation = 'h',
    )
    price_variety_box.append(obj)
    
layout = go.Layout(
    title='Price distribution in varieties',
    autosize=False,
    width=800,
    height=400,
    xaxis=dict(autorange=False,range=[4, 150])
)

fig = go.Figure(data=price_variety_box, layout=layout)
py.iplot(fig)


points_variety_box = []

for variety in wine_variety_15.index:
    obj = go.Box(
        x=wines[(wines['variety'] == variety)]['points'],
        name=variety,
        orientation = 'h',
    )
    points_variety_box.append(obj)
    
layout = go.Layout(
    title='Points distribution in varieties',
    autosize=False,
    width=800,
    height=400
)

fig = go.Figure(data=points_variety_box, layout=layout)
py.iplot(fig)
wines_country_median = wines.groupby('country').median().sort_values('taste/price', ascending=False)[:15]
texts = []
for country in wines_country_median.index:
    text = 'Median points: {0}<br>Median price: {1}'.format(wines_country_median.loc[country, 'points'],
                                              wines_country_median.loc[country, 'price'])
    texts.append(text)

data = [
    go.Bar(x=wines_country_median.index, y=wines_country_median['taste/price'],
           textfont=dict(size=16, color='#333'),
           text=texts,
           marker=dict(
               color=wines_country_median['taste/price'],
               colorscale = 'Electric',
               line=dict(
                    color='rgba(50,25,25,0.5)',
                    width=1.5)
           ))
]

annot = dict(
            x=11,
            y=9,
            xref='x',
            yref='y',
            text="Hover to see the country's median price and points",
            showarrow=False
        )

layout = go.Layout(
    autosize=False,
    width=800,
    height=400,
    title='Countries with the highest taste/price ratio',
    annotations=[annot]
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
wine_variety_15 = wines['variety'].value_counts()[:15]

columns=['Variety', 'Country', 'Median taste/price', 'Median price', 'Median points']
wine_variety_df = pd.DataFrame(columns=columns)

for i, variety in enumerate(wine_variety_15.index):
    country_data = wines[wines['variety'] == variety].groupby('country').median().sort_values('taste/price', 
                                                                                              ascending=False).iloc[0]
    
    wine_variety_df.loc[i] = [variety, country_data.name, country_data['taste/price'], 
                              country_data['price'], country_data['points']]

wine_variety_df.iloc[:, :]
wines_points_95_summary = wines[wines['points'] >= 95].groupby('country').count()
wines_points_95_summary = wines_points_95_summary[['description']]
wines_points_95_summary = wines_points_95_summary.merge(wines[wines['points'] >= 95].groupby('country').mean(), 
                                                        left_index = True, right_index = True)
wines_points_95_summary = wines_points_95_summary.rename(columns={'description': 'Number of wines', 
                                                                  'points': 'Mean taste points', 'price': 'Mean price', 
                                                                  'taste/price': 'Mean taste/price'})
wines_points_95_summary
wines_points_95 = wines[wines['points'] >= 95]
wines_price_95_countries = wines_points_95['country'].unique()
traces = []

for country in wines_price_95_countries:
    trace = go.Scatter(
        x = wines_points_95[wines_points_95['country'] == country]['price'],
        y = wines_points_95[wines_points_95['country'] == country]['points'],
        mode = 'markers',
        marker = dict(
            size = 8
        ),
        text = wines_points_95[wines_points_95['country'] == country]['title'],
        name = country
    )
    traces.append(trace)

layout = go.Layout(
    autosize=False,
    width=800,
    height=400,
    title='Wines with the highest number of taste points',
    hovermode= 'closest'
)

fig = go.Figure(data=traces, layout=layout)
py.iplot(fig)
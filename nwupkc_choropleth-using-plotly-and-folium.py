import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
import folium
df = pd.read_csv("../input/world-happiness/2017.csv")
df.head()
df.tail()
sns.distplot(df['Happiness.Score']);
iplot([go.Scatter(x=df['Economy..GDP.per.Capita.'], y=df['Happiness.Score'], mode='markers')]);
sns.pairplot(df[['Economy..GDP.per.Capita.','Family','Health..Life.Expectancy.',
                 'Freedom','Generosity','Trust..Government.Corruption.']]);
sns.jointplot(x='Economy..GDP.per.Capita.', y='Happiness.Score', data=df);
sns.jointplot(x='Economy..GDP.per.Capita.', y='Happiness.Score', data=df, kind='hex', gridsize=20);
iplot([go.Choropleth(
    locationmode='country names',
    locations=df.Country,
    z=df["Happiness.Score"]
)])
df['Country2'] = df['Country'].replace("United States", "United States of America")

state_geo = os.path.join("../input/world-countries/", 'world-countries.json')
m = folium.Map(location=[20, 0], zoom_start=3.5)
m = folium.Map(location=[48.85, 2.35], tiles="Mapbox Bright", zoom_start=2)
m.choropleth(
    geo_data=state_geo,
    name='Choropleth',
    data=df,
    columns=['Country2', 'Happiness.Score'],
    key_on='feature.properties.name',
    fill_color='YlGn',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Happiness Score'
)
folium.LayerControl().add_to(m)
m
import plotly.offline as offline
import plotly.graph_objs as go
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print(os.listdir("../input"))
offline.init_notebook_mode(connected=True)
df = pd.read_csv("../input/countries of the world.csv")
df.head()
comma_cols = ['Pop. Density (per sq. mi.)', 'Coastline (coast/area ratio)',
       'Net migration', 'Infant mortality (per 1000 births)','Literacy (%)', 'Phones (per 1000)', 'Arable (%)',
       'Crops (%)', 'Other (%)', 'Climate', 'Birthrate', 'Deathrate',
       'Agriculture', 'Industry', 'Service']
for i in comma_cols:
    if df[i].dtype == 'O':
        df[i] = df[i].str.replace(',', '.')
        df[i] = df[i].astype("float64")
        
df = df.fillna(0)
df_lc_1 = df.iloc[0:12,:]

trace_0 = go.Scatter(
        
    x = df_lc_1['Country'],
    y = ((df_lc_1['GDP ($ per capita)'] - np.mean(df_lc_1['GDP ($ per capita)'])) / np.std(df_lc_1['GDP ($ per capita)'])),
    
    name = "Line Chart of GDP of first 12 countries",
    
    
    line = dict(color = ('rgb(0, 250, 24)'),
               width = 4)
)

trace_1 = go.Scatter(
        
    x = df_lc_1['Country'],
    y = ((df_lc_1['Literacy (%)'] - np.mean(df_lc_1['Literacy (%)'])) / np.std(df_lc_1['Literacy (%)'])),
    
    name = "Line Chart of Literacy of first 12 countries",
    
    
    line = dict(color = ('rgb(205, 12, 24)'),
               width = 4,
               dash = 'dot')
)

trace_2 = go.Scatter(
        
    x = df_lc_1['Country'],
    y = ((df_lc_1['Population'] - np.mean(df_lc_1['Population'])) / np.std(df_lc_1['Population'])),
    
    name = "Line Chart of Literacy of first 12 countries",
    
    
    line = dict(color = ('rgb(2, 12, 240, 191)'),
               width = 4,
               dash = 'dash')
)

data = [trace_0, trace_1, trace_2]

layout = dict(title = "Comparison for Normalized GDP, Literacy and Population for 12 countries",
             xaxis = dict(title='Year'),
             yaxis = dict(title = 'NORMALIZED: GDP, Literacy and Population'))

fig = dict(data=data, layout=layout)

offline.iplot(fig)
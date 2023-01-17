import pandas as pd
import numpy as np
import seaborn as sns
import plotly.offline as py
import matplotlib.pyplot as plt
% matplotlib inline

df = pd.read_csv('../input/honeyproduction.csv')

df.head()
df.info()
state_total = df.groupby('state')['totalprod'].sum()

state_total = state_total.to_frame().reset_index()
plt.figure(figsize=(25,8))
sns.barplot(x="state", y='totalprod', data=state_total);
state_year_totals = df.groupby(['state','year'])['totalprod'].sum().reset_index()
plt.figure(figsize=(25,8))
sns.barplot(x="state", y="totalprod", hue="year", data=state_year_totals);
state_year_Yieldtotals = df.groupby(['state','year'])['yieldpercol'].sum().reset_index()
plt.figure(figsize=(25,8))
sns.barplot(x="state", y="yieldpercol", hue="year", data=state_year_Yieldtotals);
averageprice_dict = {}

df_year = 1998
while df_year <= 2012:
    price_sum = df.query('year == @df_year')['priceperlb'].sum()
    average = price_sum / len(df.query('year == @df_year')['priceperlb'])
    averageprice_dict[df_year] = round(average, 2)
    df_year = df_year + 1
averageprice_dict
priceperlb_yearly_avg = pd.DataFrame(list(averageprice_dict.items()), columns=['year', 'average'])
sns.set_style("darkgrid")
plt.figure(figsize=(20,8))
plt.plot(priceperlb_yearly_avg['year'], priceperlb_yearly_avg['average'])
plt.title("Average Price of Honey per Pound 1998-2012", fontsize=24)
plt.ylabel('Price in USD per lb')
plt.show()
state_total2 = state_total.copy()
import plotly.plotly as py
import plotly.offline as py
py.init_notebook_mode(connected=True)

for col in state_total2.columns:
    state_total2[col] = state_total2[col].astype(str)

scl = [[0.0, 'rgb(255,235,143)'],[0.2, 'rgb(255,235,193)'],[0.4, 'rgb(255,235,119)'],\
            [0.6, 'rgb(255,220,67)'],[0.8, 'rgb(248,186,36)'],[1.0, 'rgb(216,155,39)']]

labels = state_total2['state']
values = state_total2['totalprod']

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = labels,
        z = np.array(values).astype(float),
        locationmode = 'USA-states',
        text = labels,
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Honey Production (lbs)")
        ) ]

layout = dict(
        title = 'Honey Production Since 1998<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict(data=data, layout=layout )
py.iplot(fig, filename='d3-cloropleth-map' )

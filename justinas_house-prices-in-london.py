# Load libraries



import pandas as pd

import numpy as np



import missingno as msno

import statsmodels.api as sm



from fbprophet import Prophet



import geopandas as gpd



import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go

import plotly.express as px

from pylab import rcParams

rcParams['figure.figsize'] = 15, 12



# Allows to display all of the outputs of a cell

from IPython.display import display



# Set float data type format

pd.options.display.float_format = '{:,.2f}'.format



# Set the maximum number of row to be displayed

pd.options.display.max_rows = 999



# Set global visualisation settings

plt.rc('font', size=14)   
df = pd.read_csv('../input/housing-in-london/housing_in_london_monthly_variables.csv')

df_1 = pd.read_csv('../input/housing-in-london/housing_in_london_yearly_variables.csv')



display(df.head())

display(df_1.head())
display(df.describe());



msno.matrix(df);
display(df_1.describe())



msno.matrix(df_1);
# Set date as index for easier manipulation

df = df.set_index(pd.to_datetime(df['date']))

df_1 = df_1.set_index(pd.to_datetime(df_1['date']))



del df['date']

del df_1['date']



df_1['mean_salary'] = df_1['mean_salary'].replace('-', np.NaN)

df_1['mean_salary'] = df_1['mean_salary'].replace('#', np.NaN)

df_1['mean_salary'] = df_1['mean_salary'].astype(float)



# Create dataset cuts

prices_london = df[df['borough_flag'] == 1]

prices_england = df[df['area'] == 'england']

prices_north_east = df[df['area'] == 'north east']



# Calcualte mean prices for the different cuts of data

london_mean_price = prices_london.groupby('date')['average_price'].mean()

england_mean_price = prices_england.groupby('date')['average_price'].mean()

north_east_mean_price = prices_north_east.groupby('date')['average_price'].mean()



print('Processing Complete')
fig = go.Figure()



fig.add_trace(go.Scatter(x=london_mean_price.index, 

                         y=london_mean_price.values,

                         mode='lines',

                         name='London Mean House Price',

                        ))



fig.add_trace(go.Scatter(x=england_mean_price.index, 

                         y=england_mean_price.values,

                         mode='lines',

                         name='England Mean House Price',

                        ))



fig.add_trace(go.Scatter(x=north_east_mean_price.index, 

                         y=north_east_mean_price.values,

                         mode='lines',

                         name='North East Mean House Price',

                        ))





fig.update_layout(

    template='gridon',

    title='Average Monthly House Price',

    xaxis_title='Year',

    yaxis_title='Price (£)',

    xaxis_showgrid=False,

    yaxis_showgrid=False,

    legend=dict(y=-.2, orientation='h'),

    shapes=[

        dict(

            type="line",

            x0='2016-06-01',

            x1='2016-06-01',

            y0=0,

            y1=london_mean_price.values.max()*1.2,

            line=dict(

            color="LightSalmon",

            dash="dashdot"

            )

        ),

        dict(

            type="rect",

            x0="2007-12-01",

            y0=0,

            x1="2009-06-01",

            y1=london_mean_price.values.max()*1.2,

            fillcolor="LightSalmon",

            opacity=0.5,

            layer="below",

            line_width=0,

        ),

        dict(

            type="rect",

            x0="2001-03-01",

            y0=0,

            x1="2001-11-01",

            y1=london_mean_price.values.max()*1.2,

            fillcolor="LightSalmon",

            opacity=0.5,

            layer="below",

            line_width=0,

        )

    ],

    annotations=[

            dict(text="The Great Recession", x='2007-12-01', y=london_mean_price.values.max()*1.2),

            dict(text="Brexit Vote", x='2016-06-01', y=london_mean_price.values.max()*1.2),

            dict(text="Dot-Com Bubble Recession", x='2001-03-01', y=london_mean_price.values.max()*1.2)

    ]

)



fig.show()
fig = px.line(prices_london, x=prices_london.index, y="average_price", color='area')



fig.update_layout(

    template='gridon',

    title='Average Monthly London House Price by Borough',

    xaxis_title='Year',

    yaxis_title='Price (£)',

    xaxis_showgrid=False,

    yaxis_showgrid=False

)



fig.show()
# Calculate the mean yearly price per borough

yearly_prices_london = prices_london.groupby('area').resample('y')['average_price'].mean()



# Calculate the yealy average price percentage change

yearly_prices_london_pct_ch = yearly_prices_london.groupby(level='area').apply(lambda x: x.pct_change())



yearly_prices_london_pct_ch = yearly_prices_london_pct_ch.unstack()

yearly_prices_london_pct_ch = yearly_prices_london_pct_ch.iloc[::-1]



del yearly_prices_london_pct_ch['1995-12-31']
fig = go.Figure(data=go.Heatmap(

        z=yearly_prices_london_pct_ch.values,

        x=yearly_prices_london_pct_ch.columns,

        y=yearly_prices_london_pct_ch.index,

        colorscale='Cividis'))



fig.update_layout(

    title='YoY Average London House Price Percentage Change',

    title_x=0.5,

    yaxis_nticks=33,

    xaxis_title='Year',

    yaxis_title='Borough'

)



fig.show()
# Load the shape file for mapping

map_df = gpd.read_file('../input/london-borough-and-ward-boundaries-up-to-2014/London_Wards/Boroughs/London_Borough_Excluding_MHW.shp')



map_df = map_df[['NAME', 'geometry']]

map_df = map_df.rename(columns={'NAME': 'area'})

map_df['area'] = map_df['area'].str.lower()

map_df['area'] = map_df['area'].str.replace('&','and')



yearly_prices_london_df_map = pd.DataFrame(yearly_prices_london)

yearly_prices_london_df_map = yearly_prices_london_df_map.reset_index()



price_map = map_df.set_index('area').join(yearly_prices_london_df_map.set_index('area'))

price_map = price_map.reset_index()



price_map = price_map[price_map['date'] == '2019-12-31']
variable = 'average_price'



fig, ax = plt.subplots(1, figsize=(15, 10))

price_map.plot(column=variable, 

               cmap='Blues', 

               linewidth=1, 

               ax=ax, 

               edgecolor='0.8', 

               legend=True,

               legend_kwds={'label': "Average House Price",

                           'orientation': "horizontal"})

ax.axis('off')

plt.suptitle('Average London House Price by Borough Visualised')

plt.show()
# The code below is an atempt to create an interactive map visualisation. No success so far



"""

price_map = price_map[:100]



price_map.crs = {'init': 'epsg:4326'}



price_map.to_file("output.json", driver="GeoJSON")



json_file_path = "output.json"



with open(json_file_path, 'r') as j:

     contents = json.loads(j.read())

        

fig = px.choropleth_mapbox(price_map, geojson=contents, color="average_price", locations='area', featureidkey="properties.area",

                           mapbox_style="carto-positron", zoom=9)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()

"""
london_mean_values = prices_london.groupby('area').resample('y').mean().reset_index()

df_1 = df_1.groupby('area').resample('y').mean().reset_index()

london_mean_values = pd.merge(london_mean_values, df_1, on=['area', 'date'], how='left')



del london_mean_values['borough_flag_x']

del london_mean_values['borough_flag_y']
# Compute the correlation matrix

corr = london_mean_values.corr()



# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=np.bool))



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(250, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
sns.pairplot(london_mean_values, corner=True, kind='reg', diag_kind='kde', plot_kws=dict(scatter_kws=dict(s=2)))

plt.show()
decomposition = sm.tsa.seasonal_decompose(london_mean_price, model='additive')

fig = decomposition.plot()

plt.show()
# Prepare the 

model_df = pd.DataFrame(london_mean_price).reset_index()

model_df = model_df.rename(columns={'date': 'ds', 'average_price': 'y'})



# Initialise the model and make predictions

m = Prophet()

m.fit(model_df)



future = m.make_future_dataframe(periods=24, freq='M')



forecast = m.predict(future)



# Visualise the prediction

fig1 = m.plot(forecast)
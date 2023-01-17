import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px



import os

print(os.listdir('../input/military-expenditure-of-countries-19602019'))
data = pd.read_csv('../input/military-expenditure-of-countries-19602019/Military Expenditure.csv')

data.head()
import missingno as msno

data2 = msno.nullity_sort(data, sort='descending')

msno.matrix(data2, color=(0.294, 0.325, 0.125)) #color : Army 
import plotly.graph_objects as go



fig = go.Figure(data=[go.Table(cells=dict(values=data['Name'].values.reshape(12, 22)))])

fig.update_layout(title=f'Countries Name List ({data.shape[0]})')

fig.show()
print(f"There are {len(data['Indicator Name'].unique())} types of indicator in this dataset.")
data.drop(['Indicator Name'], axis=1, inplace=True)

data.head()
third_data = px.data.gapminder()

third_data.head()
code_continent = third_data[['iso_alpha', 'continent']].drop_duplicates()

code_continent.rename(columns={'iso_alpha':'Code'}, inplace=True)

code_continent.head()
clean_data = pd.merge(data, code_continent , how='left')

clean_data.head()
clean_data['continent'] = clean_data['continent'].fillna('unknown')

clean_data = clean_data.fillna(0)
fig = px.histogram(clean_data, x="Type", y="Type", color="Type")

fig.show()
fig = px.histogram(clean_data, x="continent", y="continent", color="continent")

fig.update_layout(title='Continent Distribution')

fig.show()
data_time = clean_data.melt(id_vars=['Name', 'Code', 'Type', 'continent'])

data_time.rename(columns={'variable' : 'year'}, inplace=True)

data_time
msno.matrix(data_time, color=(0.294, 0.325, 0.125))
# type 1 : default graph

# location : country code 

# hover_name : hover information, single column or columns list

import plotly.express as px



fig = px.scatter_geo(clean_data, 

                     locations="Code",

                     hover_name="Name",

                    )

fig.update_layout(title="Simple Map")

fig.show()
# type 2 : projection type change



fig = px.scatter_geo(

    clean_data, 

    locations = 'Code',

    hover_name="Name",

    projection="orthographic",

)



fig.update_layout(title='Orthographic Earth')



fig.show()

# type 2-2 : projection type change



fig = px.scatter_geo(

    clean_data, 

    locations = 'Code',

    hover_name="Name",

    projection="natural earth",

)



fig.update_layout(title='Natural Earth')



fig.show()

# type 3 : change scatter marker size

# size : marker size



fig = px.scatter_geo(

    clean_data, 

    locations = 'Code',

    hover_name="Name",

    size = '2018',

)



fig.show()
fig, ax = plt.subplots(1, 2,figsize=(20, 7))

sns.distplot(clean_data['2018'], color='orange', ax=ax[0])

sns.boxplot(y=clean_data['2018'], color='orange', ax=ax[1])

ax[1].set_title("50% of 2018 non-zero data is under {}".format(clean_data[clean_data['2018'] > 0]['2018'].quantile(0.5)))

plt.show()
# type 3-2 : remove outliers



fig = px.scatter_geo(

    clean_data[clean_data['2018'] < 0.3 * 1e12 ], 

    locations = 'Code',

    hover_name="Name",

    size = '2018',

    projection="natural earth",

)



fig.show()
# type 4 : color



fig = px.scatter_geo(

    clean_data, 

    locations = 'Code',

    hover_name="Name",

    color = 'continent'

)



fig.show()
# type 4-2 : color with size



fig = px.scatter_geo(

    clean_data[clean_data['2018'] < 0.2 * 1e12 ], 

    locations = 'Code',

    hover_name="Name",

    size='2018',

    color = 'continent'

)



fig.show()
# type 5 : animation with year

fig = px.scatter_geo(data_time, locations="Code", color="continent",

                     hover_name="Name", size="value",

                     animation_frame="year",

                     projection="natural earth")



fig.update_layout(title='Animation but USA...')

fig.show()

# type 5-2 : animation with default data

# this is population dataset

gapminder = px.data.gapminder()

fig = px.scatter_geo(gapminder, locations="iso_alpha", color="continent",

                     hover_name="country", size="pop",

                     animation_frame="year", 

                     projection="natural earth")

fig.show()

# type 1 : default choropleth

fig = px.choropleth(clean_data, locations="Code", color="2018",

                     hover_name="Name", 

                    range_color=[0,10000000000],

                     projection="natural earth")



fig.update_layout(title='Choropleth Graph')

fig.show()

# type 2 : choropleth with animation



fig = px.choropleth(data_time, locations="Code", color="value",

                     hover_name="Name", 

                    range_color=[0,1000000000],

                    animation_frame='year')



fig.update_layout(title='Choropleth Graph Animation')

fig.show()

# type 1 : default

fig = px.line_geo(clean_data[clean_data['continent'] !='unknown'], 

                  locations="Code", 

                  color="continent")

fig.show()
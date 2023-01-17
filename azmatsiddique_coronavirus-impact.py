import pandas as pd

import plotly.offline as py

import matplotlib.pyplot as plt

data_cleaned = pd.read_csv('/kaggle/input/2019-coronavirus-dataset-01212020-01262020/2019_nC0v_20200121_20200126_cleaned.csv').drop('Unnamed: 0',axis = 1)

data_summary = pd.read_csv('/kaggle/input/2019-coronavirus-dataset-01212020-01262020/2019_nC0v_20200121_20200126 - SUMMARY.csv')

data_1  = pd.read_csv('/kaggle/input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200128.csv')

data_2 = pd.read_csv('/kaggle/input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200131.csv')

data_3 = pd.read_csv('/kaggle/input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200127.csv')

data_4 = pd.read_csv('/kaggle/input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200131.csv')
data_1.head()
data_2.head()


new_data = pd.concat([data_1, data_2,data_3,data_4], ignore_index=True)

new_data['Last Update'] = pd.to_datetime(new_data['Last Update'])
import pandas_profiling as npp

profile = npp.ProfileReport(new_data)

profile
new_data['Country/Region'].replace('Mainland China', 'China', inplace=True)

new_data['Country/Region'].replace('Hong Kong', 'China', inplace=True)

new_data['Country/Region'].replace('Macau', 'China', inplace=True)

new_data['Country/Region'].replace('United States', 'United States of America', inplace=True)

new_data['Country/Region'].replace('US', 'United States of America', inplace=True)

new_data['Country/Region'].replace('Singapore', 'Malaysia', inplace=True) 

new_data['Country/Region'].replace('Ivory Coast', "Côte d'Ivoire", inplace=True)
group_x = new_data.groupby(["Province/State", "Country/Region"])
group_x['Confirmed'].sum().head()

group_x['Death'].sum().head()
a = new_data.stack()
a.to_frame(name='result') #last updated data


data_country = new_data.groupby('Country/Region')['Confirmed'].sum()



worldmap = [dict(type = 'choropleth', locations = data_country.index, locationmode = 'country names',

                 z = data_country.values, colorscale = "Inferno", reversescale = True, 

                 marker = dict(line = dict( width = 0.2)), 

                 colorbar = dict(autotick = False, title = 'Number of Confirmed cases'))]



layout = dict(title = 'Coronavirus across world', geo = dict(showframe = True, showcoastlines = True, 

                                                                projection = dict(type = 'Mercator')))



fig = dict(data=worldmap, layout=layout)

py.iplot(fig, validate=False)
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly_express as px

import plotly.graph_objects as go

from IPython.display import Image

from sklearn.linear_model import LinearRegression

from scipy import stats
big_mac_countries = ['ARG', 'AUS', 'BRA', 'GBR', 'CAN', 'CHL', 'CHN', 'CZE', 'DNK',

                      'EGY', 'HKG', 'HUN', 'IDN', 'ISR', 'JPN', 'MYS', 'MEX', 'NZL',

                      'NOR', 'PER', 'PHL', 'POL', 'RUS', 'SAU', 'SGP', 'ZAF', 'KOR',

                      'SWE', 'CHE', 'TWN', 'THA', 'TUR', 'ARE', 'USA', 'COL', 'CRI',

                      'PAK', 'LKA', 'UKR', 'URY', 'IND', 'VNM', 'GTM', 'HND',

                      'NIC', 'AZE', 'BHR', 'HRV', 'JOR', 'KWT', 'LBN', 'MDA', 'OMN',

                      'QAT', 'ROU', 'EUZ']
big_mac = pd.read_csv('../input/big-mac-source-data.csv')

big_mac.tail()
big_mac = big_mac[big_mac["iso_a3"].isin(big_mac_countries)]

big_mac.info()
big_mac["dollar_price"] = big_mac["local_price"] / big_mac["dollar_ex"]

big_mac.tail()
base_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CNY']
big_mac_2019 = big_mac[big_mac['date']==big_mac['date'].max()]
for currency in base_currencies: 

    currency_price = float(big_mac_2019.loc[big_mac_2019["currency_code"]== currency, "dollar_price"])

    big_mac_2019[currency] = big_mac_2019["dollar_price"] / currency_price - 1
big_mac_2019 = big_mac_2019.round(decimals = 3)

big_mac_2019.tail()
fig = px.bar(big_mac_2019, x="USD", y="name", orientation='h', color="USD").update_yaxes(categoryorder="total ascending")

fig.show()
fig = px.choropleth(big_mac_2019, locations="iso_a3",

                    color="USD", # lifeExp is a column of gapminder

                    hover_name="name", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Plasma)

fig.show()
big_mac_gdp_data = big_mac_2019[big_mac_2019["GDP_dollar"] > 0]

big_mac_gdp_data.tail()
slope, intercept, r_value, p_value, std_err = stats.linregress(big_mac_gdp_data["GDP_dollar"], big_mac_gdp_data["dollar_price"])

# Create regression line

regressLine = intercept + big_mac_gdp_data["GDP_dollar"]*slope

plt.clf()
fig = px.scatter(big_mac_gdp_data, x="GDP_dollar", y="dollar_price",

                 size='GDP_dollar', hover_data=['name'])

fig.add_trace(go.Line(y = regressLine, x =big_mac_gdp_data["GDP_dollar"], name = 'RegressLine'))



fig.update_layout(title='Big Mac prices v GDP per person',

                   xaxis_title='GDP per person ($)',

                   yaxis_title='Big Mac price ($)')



fig.show()
big_mac_gdp_data["adj_price"] = regressLine

big_mac_adj_index = big_mac_gdp_data[["name","iso_a3","currency_code","dollar_price","adj_price","dollar_ex","GDP_dollar","date"]]

big_mac_adj_index.tail()
for currency in base_currencies: 

    dollar_currency = float(big_mac_adj_index.loc[big_mac_adj_index["currency_code"]== currency, "dollar_price"])

    currency_price = float(big_mac_adj_index.loc[big_mac_adj_index["currency_code"]== currency, "adj_price"])

    big_mac_adj_index[currency] = (big_mac_adj_index["dollar_price"] / big_mac_adj_index["adj_price"]) / (dollar_currency / currency_price) - 1

big_mac_adj_index.tail()
regression_countries = ['ARG', 'AUS', 'BRA', 'GBR', 'CAN', 'CHL', 'CHN', 'CZE', 'DNK',

                         'EGY', 'EUZ', 'HKG', 'HUN', 'IDN', 'ISR', 'JPN', 'MYS', 'MEX',

                         'NZL', 'NOR', 'PER', 'PHL', 'POL', 'RUS', 'SAU', 'SGP', 'ZAF',

                         'KOR', 'SWE', 'CHE', 'TWN', 'THA', 'TUR', 'USA', 'COL', 'PAK',

                         'IND', 'AUT', 'BEL', 'NLD', 'FIN', 'FRA', 'DEU', 'IRL', 'ITA',

                         'PRT', 'ESP', 'GRC', 'EST']
big_mac_adj_index = big_mac_adj_index[big_mac_adj_index["iso_a3"].isin(regression_countries)]

fig = px.bar(big_mac_adj_index, x="USD", y="name", orientation='h', color="USD").update_yaxes(categoryorder="total ascending")

fig.show()
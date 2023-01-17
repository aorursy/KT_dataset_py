import pandas as pd
data = pd.read_csv('/kaggle/input/india-trade-data/2018-2010_import.csv')
data.head()
data['Commodity'] = data['Commodity'].map(lambda x: x[:15])
data.head(2)
highest_imported_country = data.groupby(['country'])['value'].sum()
highest_imported_country.sort_values(ascending=False)
## get the percentage share

highest_imported_country = highest_imported_country.apply(lambda x:round(100 * x/highest_imported_country.sum())).sort_values(ascending=False)
## cut short the data to only top 10 countries

highest_imported_country = highest_imported_country[:10]
highest_imported_country
highest_imported_commodity = data.groupby('Commodity')['value'].sum()
highest_imported_commodity.sort_values(ascending=False)
## getting the percentage share

highest_imported_commodity = highest_imported_commodity.apply(lambda x:round(100 * x/highest_imported_commodity.sum())).sort_values(ascending=False)
## get the top 10 commodities only

highest_imported_commodity = highest_imported_commodity[:10]
highest_imported_commodity
top_10_countries = highest_imported_country.index.tolist()
countries_commodity_data = data.groupby(['country', 'Commodity'])['value'].sum()
countries_commodity_data
countries_commodity_data.groupby(level=1).sum()
sorted_country_commodity_data = countries_commodity_data.groupby(level=0).apply(lambda x: round(100 * x/x.sum())).sort_values(ascending=False)
sorted_country_commodity_data.loc['CHINA P RP']
sorted_country_commodity_data = sorted_country_commodity_data.groupby(level=0).head()
sorted_country_commodity_data.loc['CHINA P RP']
sorted_country_commodity_data.head(2)
top_10_countries
## This command is working.. need to check to get the multiindex for all the top countries

# sorted_country_commodity_data.loc[sorted_country_commodity_data.index.get_level_values('country') == 'SAUDI ARAB']



sorted_country_commodity_data.loc[sorted_country_commodity_data.index.get_level_values('country') == 'SAUDI ARAB']
sorted_country_commodity_data.loc['SAUDI ARAB']
for country in top_10_countries:

    print(country)

    print(sorted_country_commodity_data.loc[country])
labels_list = top_10_countries.copy()
highest_imported_country.loc['U S A']
labels_list = []

values_list = []

for country in top_10_countries:

    ## 1. add the percentage value of country

    values_list.append(highest_imported_country.loc[country])

    ## 2. add the percentage value of commodities

    minerals = sorted_country_commodity_data.loc[country].index.tolist()

    minerals_values = sorted_country_commodity_data.loc[country].values.tolist()

    for i in range(0, len(minerals)):

        minerals[i] = country + "-" + minerals[i]

    labels_list.append(country)

    labels_list.extend(minerals)

    values_list.extend(minerals_values)
print("Labels count:", len(labels_list))

print("Values count:", len(values_list))
parent_list = []

current_country = ""

i = 0

for country in labels_list:

    if (country in top_10_countries):

        parent_list.append("Commodities")

        current_country = country

#         values_list.append(i)

    else:

        parent_list.append(current_country)

    i = i+1
len(parent_list)
import matplotlib.pyplot as plt

import plotly.graph_objects as go
trace = go.Sunburst(

    labels = labels_list,

    parents = parent_list,

    values = values_list,

#     branchvalues="total",

    outsidetextfont = {"size": 20, "color": "#377eb8"},

    marker = {"line": {"width": 2}},

)



layout = go.Layout(

    margin = go.layout.Margin(t=0, l=0, r=0, b=0)

)



go.Figure([trace], layout).show()
top_10_commodities = highest_imported_commodity.index.tolist()
top_10_commodities
commodity_countries_data = data.groupby(['Commodity', 'country'])['value'].sum()
commodity_countries_data
commodity_countries_data.groupby(level=1).sum()
sorted_commodity_countries_data = commodity_countries_data.groupby(level=0).apply(lambda x: round(100 * x/x.sum())).sort_values(ascending=False)
sorted_commodity_countries_data.loc['MINERAL FUELS, ']
sorted_commodity_countries_data = sorted_commodity_countries_data.groupby(level=0).head()
sorted_commodity_countries_data.loc['MINERAL FUELS, ']
sorted_commodity_countries_data.head(2)
top_10_commodities
## This command is working.. need to check to get the multiindex for all the top countries

# sorted_commodity_countries_data.loc[sorted_commodity_countries_data.index.get_level_values('Commodity') == 'MINERAL FUELS, ']



sorted_commodity_countries_data.loc[sorted_commodity_countries_data.index.get_level_values('Commodity') == 'MINERAL FUELS, ']
sorted_commodity_countries_data.loc['MINERAL FUELS, ']
for commodity in top_10_commodities:

    print(commodity)

    print(sorted_commodity_countries_data.loc[commodity])
labels_list_commodities = top_10_commodities.copy()
highest_imported_commodity.loc['ANIMAL OR VEGET']
labels_list_commodities = []

values_list_commodities = []

for commodity in top_10_commodities:

    ## 1. add the percentage value of country

    values_list_commodities.append(highest_imported_commodity.loc[commodity])

    ## 2. add the percentage value of commodities

    countries = sorted_commodity_countries_data.loc[commodity].index.tolist()

    countries_values = sorted_commodity_countries_data.loc[commodity].values.tolist()

    for i in range(0, len(countries)):

        countries[i] = countries[i] + "-" + commodity

    labels_list_commodities.append(commodity)

    labels_list_commodities.extend(countries)

    values_list_commodities.extend(countries_values)
print("Labels count:", len(labels_list_commodities))

print("Values count:", len(values_list_commodities))
parent_list_commodities = []

current_commodity = ""

i = 0

for commodity in labels_list_commodities:

    if (commodity in top_10_commodities):

        parent_list_commodities.append("Commodities")

        current_commodity = commodity

#         values_list.append(i)

    else:

        parent_list_commodities.append(current_commodity)

    i = i+1
len(parent_list_commodities)
import matplotlib.pyplot as plt

import plotly.graph_objects as go
trace = go.Sunburst(

    labels = labels_list_commodities,

    parents = parent_list_commodities,

    values = values_list_commodities,

#     branchvalues="total",

    outsidetextfont = {"size": 20, "color": "#377eb8"},

    marker = {"line": {"width": 2}},

)



layout = go.Layout(

    margin = go.layout.Margin(t=0, l=0, r=0, b=0)

)



go.Figure([trace], layout).show()
top_10_commodities
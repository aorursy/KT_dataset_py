import pandas as pd

import plotly.express as px

import plotly.graph_objects as go

import numpy as np

import datetime

from pathlib import Path
PATH = Path('../input/covid19-global-forecasting-week-2')

list(PATH.glob('*'))
data = pd.read_csv(PATH/'train.csv')

data.head()
agg_funcs = {'Date': 'first', 'ConfirmedCases': 'sum', 'Fatalities': 'sum'}

data_sum = data.groupby(data['Date']).aggregate(agg_funcs)

data_sum
fig = go.Figure()

fig.add_trace(go.Scatter(x=data_sum['Date'], y=data_sum['ConfirmedCases'], mode='lines', name='Confirmed Cases'))

fig.add_trace(go.Scatter(x=data_sum['Date'], y=data_sum['Fatalities'], mode='lines', name='Fatalities'))

fig.show()
countries = list(set(list(data['Country_Region'])))

print(countries)
num_cases = []

for c, country in enumerate(countries):

    data2 = data.loc[data['Country_Region'] == country]

    num_cases_country = data2.groupby(data2['Date']).aggregate(agg_funcs).max().ConfirmedCases

    num_cases.append(num_cases_country)



# index ordered by num_cases    

idx_top_by_cases = list(reversed(np.argsort(num_cases)))



for i in range(20):

    idx_top = idx_top_by_cases[i]

    print('%d: %s (%d cases)' % (i+1, countries[idx_top], num_cases[idx_top]))
countries_str = '[%s]'% (', '.join(["'%s'"%countries[idx] for idx in idx_top_by_cases[:20]]))   # there must be a less ugly way to do this in pandas

data_top_countries = data.query("Country_Region == %s" % countries_str) 



fig = px.line(data_top_countries, x="Date", y="ConfirmedCases", color="Country_Region",

              line_group="Country_Region", hover_name="Country_Region",

              title="Daily cases for top 20 countries (with range slider)")

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()
num_fatalities = []

for c, country in enumerate(countries):

    data2 = data.loc[data['Country_Region'] == country]

    num_fatalities_country = data2.groupby(data2['Date']).aggregate(agg_funcs).max().Fatalities

    num_fatalities.append(num_fatalities_country)



# index ordered by num_cases    

idx_top_by_fatalities = list(reversed(np.argsort(num_fatalities)))



for i in range(20):

    idx_top = idx_top_by_fatalities[i]

    print('%d: %s (%d fatalities)' % (i+1, countries[idx_top], num_fatalities[idx_top]))
countries_str = '[%s]'% (', '.join(["'%s'"%countries[idx] for idx in idx_top_by_fatalities[:20]]))   # there must be a less ugly way to do this in pandas

data_top_countries = data.query("Country_Region == %s" % countries_str) 



fig = px.line(data_top_countries, x="Date", y="Fatalities", color="Country_Region",

              line_group="Country_Region", hover_name="Country_Region",

              title="Daily fatalities for top 20 countries (with range slider)")

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()
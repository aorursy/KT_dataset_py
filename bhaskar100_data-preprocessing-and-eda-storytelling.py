import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import plotly

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots



import os

path = '/kaggle/input/ntt-data-global-ai-challenge-06-2020/'

        

import re

import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

co_trn = pd.read_csv(path + 'COVID-19_train.csv')

co_tst = pd.read_csv(path + 'COVID-19_test.csv')

cr_oil = pd.read_csv(path + 'Crude_oil_trend_From1986-10-16_To2020-03-31.csv')
# Setting index as Date and printing the head

co_trn.set_index('Date', inplace = True)

co_trn.head()
# We can also see which countires have more number of cases in column wise

co_trn_sorted = co_trn.loc[:,co_trn.max().sort_values(ascending = False).index]

co_trn_sorted.head()
# Selecting only total_cases and total_deaths from the dataframe

countries = co_trn.columns.tolist()



# total_cases = co_trn.loc[:,countries[::4]]    Use this for total_cases

# total_deaths = co_trn.loc[:,countries[2::4]]  Use this for total_deaths



# Here we are selecting only total_cases and total_deaths of each country.

cov_tc_td = co_trn.loc[:,countries[::2]]

cov_tc_td.head()
cov_tc_td['Price'].plot(figsize = (15,4), title = 'Oil Price Index')

plt.xticks(rotation = 60);

plt.ylabel('Oil Price');
cr_oil.set_index('Date', inplace = True)

cr_oil['Price'].plot(title = 'Daily Crude Oil Index', figsize = (15,5))

plt.xticks(rotation = 60);

plt.ylabel('Oil Price');
# Create figure with secondary y-axis.

fig = make_subplots(specs = [[{'secondary_y' : True}]])



fig.add_trace(go.Scatter(x = cov_tc_td.index, y = cov_tc_td.World_total_cases,

                        mode = 'lines',

                        name = 'World Total Cases'),

             secondary_y = False)

fig.add_trace(go.Scatter(x = cov_tc_td.index, y = cov_tc_td.Price,

                        mode = 'lines',

                        name = 'Oil Price'),

             secondary_y = True)



# Adding figure title

fig.update_layout(title = '<b>World Total Cases Vs Oil Prices</b>')



# Setting X-axis title

fig.update_xaxes(title = '<b>Dates</b>')



# Setting Y-axis labels

fig.update_yaxes(title_text = '<b>Oil Price</b>', secondary_y = True)

fig.update_yaxes(title_text = '<b>World Total Cases</b>', secondary_y = False)



# Display the figure

fig.show()
# Slicing dataframe with only [country]_total_cases

total_cases = co_trn.loc[:,countries[::4]]    #Use this for total_cases



# Let's remove the Date index and convert the date to pd.to_datetime() format.

total_cases = total_cases.reset_index()

total_cases['Date'] = pd.to_datetime(total_cases['Date'])

total_cases_till_date = total_cases[total_cases.Date == total_cases.Date.max()]



# # Now Let's melt the dataframe i.e swapping columns to rows

total_cases_till_date = total_cases_till_date.melt(id_vars = 'Date', var_name = 'Countries', value_name = 'Confirmed_Cases')

total_cases_till_date.sort_values(by = 'Confirmed_Cases', ascending = False, inplace = True)

total_cases_till_date.head()
# Now Let's remove the Country names that ended with _total_cases



def try_extract(pattern, string):

    try:

        m = pattern.search(string)

        return m.group(0)

    except (TypeError, ValueError, AttributeError):

        return np.nan



p = re.compile(r'[a-zA-Z]*(?=_)')

total_cases_till_date['Countries'] = [try_extract(p, x) for x in total_cases_till_date['Countries']]

total_cases_till_date.head()
# Now We can print the top 30 countries that has recorded Confirmed_Cases till date 



# Uncomment the below lines so that you can see the bar chart of Top 30 countries of Confirmed_Cases



# fig = px.bar(total_cases_till_date[1:30], x = 'Countries', y = 'Confirmed_Cases',

#             hover_data = ['Countries', 'Confirmed_Cases'], color = 'Confirmed_Cases',

#             color_continuous_scale = px.colors.diverging.Portland,

#             title = '<b>Top 30 Countries of Confirmed Cases</b>')

# fig.show()
# Slicing the dataframe with only [country]_total_deaths

total_deaths = co_trn.loc[:, countries[2::4]]



total_deaths = total_deaths.reset_index()

total_deaths['Date'] = pd.to_datetime(total_deaths['Date'])

total_deaths_till_date = total_deaths[total_deaths['Date'] == total_deaths['Date'].max()]



# Now Let's melt the dataframe i.e swapping columns to rows

total_deaths_till_date = total_deaths_till_date.melt(id_vars = 'Date', var_name = 'Countries', value_name = 'Confirmed_Deaths')

total_deaths_till_date.sort_values(by = 'Confirmed_Deaths', ascending = False, inplace = True)

total_deaths_till_date.head()
total_deaths_till_date['Countries'] = [try_extract(p, x) for x in total_deaths_till_date['Countries']]

total_deaths_till_date.head()
# Now We can also print the top 30 countries that has recorded Confirmed_Deaths till date 



# Uncomment the below lines so that you can see the bar chart of Top 30 countries of Confirmed_Cases



# fig = px.bar(total_deaths_till_date[1:30], x = 'Countries', y = 'Confirmed_Deaths',

#             hover_data = ['Countries', 'Confirmed_Deaths'], color = 'Confirmed_Deaths',

#             color_continuous_scale = px.colors.diverging.Portland,

#             title = '<b>Top 30 Countries of Confirmed Deaths</b>')

# fig.show()
# Now merge the two dataframes of total_cases and total_deaths so that we can plot a stacked bar chart of Confirmed_Cases 

# and Confirmed_Deaths

total_cd_till_date = pd.merge(total_cases_till_date, total_deaths_till_date, on = ['Date', 'Countries'], how = 'left')

total_cd_till_date.head()
# Now Let's plot the stacked barchart for the above dataframe

fig = go.Figure(data = [

    go.Bar(name = 'Confirmed Cases',

           x = total_cd_till_date['Countries'][1:30],

           y = total_cd_till_date['Confirmed_Cases'][1:30],                     

           text = total_cd_till_date['Confirmed_Cases'][1:30],

           textposition = 'inside'),

    go.Bar(name = 'Confirmed Deaths',

           x = total_cd_till_date['Countries'][1:30],

           y = total_cd_till_date['Confirmed_Deaths'][1:30],                      

           text = total_cd_till_date['Confirmed_Deaths'][1:30],

           textposition = 'inside')

])

# Changing the bar mode

fig.update_layout(barmode = 'stack',

                  title = '<b>Stacked Bar Chart of Top 30 Countries</b>',

                  xaxis_title = '<b>Countries</b>',

                  yaxis_title = '<b>Confirmed Cases and Deaths</b>')

fig.show()
total_cases_melt = total_cases.melt(id_vars = ['Date', 'Price'], var_name = 'Countries', value_name = 'Confirmed_Cases')

total_cases_melt.sort_values(by = 'Confirmed_Cases', ascending = False)



total_cases_melt['Countries'] = [try_extract(p, x) for x in total_cases_melt['Countries']]

total_cases_melt.head()
total_deaths['Price'] = co_trn['Price'].values

total_deaths_melt = total_deaths.melt(id_vars = ['Date', 'Price'], var_name = 'Countries', value_name = 'Confirmed_Deaths')

total_deaths_melt.sort_values(by = 'Confirmed_Deaths', ascending = False)



total_deaths_melt['Countries'] = [try_extract(p, x) for x in total_deaths_melt['Countries']]

total_deaths_melt.head()
total_cd_melt = pd.merge(total_cases_melt, total_deaths_melt, on = ['Date', 'Price', 'Countries'], how = 'left')

total_cd_melt.sort_values(by = 'Confirmed_Cases', ascending = False)

total_cd_melt.head()
total_cd_melt['Date'].dtype
# Here total_cd_melt['Date'] series datatype is DateTimeIndex. To plot a geographical map with animation_frame = 'Date',

# we need to convert the DateTimeIndex to string type. The below steps are the process..



total_cd_melt['Date'] = total_cd_melt['Date'].apply(lambda x: str(x).split()[0])

total_cd_melt = total_cd_melt[total_cd_melt['Countries'] != 'World']
fig = px.scatter_geo(total_cd_melt,

                     locations = 'Countries',

                     locationmode = 'country names',

                     color = 'Confirmed_Cases',

                     hover_name = 'Countries',

                     hover_data = ['Confirmed_Deaths','Price'],

                     size = 'Confirmed_Cases',

                     animation_frame = 'Date',

                     color_continuous_scale = px.colors.diverging.Portland,

                     title = '<b>Spread of Corona Virus accross the Globe</b>',

                     projection = 'natural earth')





fig.show()
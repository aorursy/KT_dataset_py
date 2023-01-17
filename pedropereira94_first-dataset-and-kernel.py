import numpy as np

import pandas as pd 

import plotly.express as px

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Reading in file containing the population for each state 

pop = pd.read_csv('/kaggle/input/2020-population/population estimate 2020.csv')

#Renaming the 'State' column. This is necessary to merge with the other dataframe. The column names must match.

pop = pop.rename(columns={'State':'state'})

#Reading in the background checks data

firearm_data = pd.read_csv('/kaggle/input/nics-firearm-background-checks/nics-firearm-background-checks.csv')

#Merging the population data with the backgrund checks data by the common column 'state'.

firearm_data_w_pop = firearm_data.merge(pop, on = 'state')

#Retaining only the necessary columns from the 2 data frames created above

firearm_data = firearm_data[['month', 'state', 'totals']]

firearm_data_w_pop = firearm_data_w_pop[['month', 'state', 'totals', 'Pop']]

#Renaming the month column to date for both data frames 

firearm_data = firearm_data.rename(columns={'month':'date'})

firearm_data_w_pop = firearm_data_w_pop.rename(columns={'month':'date'})

#Converting the date column to datetime

firearm_data['date']= pd.to_datetime(firearm_data['date'])

firearm_data_w_pop['date']= pd.to_datetime(firearm_data_w_pop['date'])

#Setting the date as the index for both data frames

firearm_data = firearm_data.set_index('date')

firearm_data_w_pop = firearm_data_w_pop.set_index('date')

print(firearm_data.head())

print('\n')

print(firearm_data_w_pop.head())
#Resampling the data on an annual basis and adding all the intances in that year together

yearly = firearm_data.resample('A').sum()

#Displaying new resampled data

yearly.head()
#Calculating the yearly percentage growth of total permits in the country and adding to a new column

yearly['pct_growth'] = yearly.pct_change() * 100

#Rounding the numnber to 2 decimal places

yearly = yearly.round(2)

#Slicing the data to remove the 2020 data as it is impartial and can throw off the overall trend analysis

yearly = yearly[:'20191231']

#Ensuring that 2020 data is not included

yearly.tail()
#Creating visualization of the total numbers of yearly background checks since the program began

fig = px.bar(yearly, y='totals', x = yearly.index.year, color_discrete_sequence=px.colors.qualitative.Dark2)

fig.update_layout(

    title="NICS Background checks from 1998 to 2019",

    xaxis_title="Year",

    yaxis_title='Background checks (in millions)')

fig.show()
#Creating visualization of the percentage growth in brackground checks year over year

fig2 = px.line(yearly, y='pct_growth', x = yearly.index.year, color_discrete_sequence=px.colors.qualitative.Bold)

fig2.update_layout(

    title='NICS Background checks yearly % change',

    xaxis_title="Year",

    yaxis_title='% Change')

fig2.show()
#Slicing data to omitt the 1999 percentage growth 

yearly_2000s = yearly['2000-12-31':]

#Creating visualization of the percentage growth in brackground checks year over year excluding 1999

fig3 = px.line(yearly_2000s, y='pct_growth', x = yearly_2000s.index.year, color_discrete_sequence=px.colors.qualitative.Set1)

fig3.update_layout(

    title='NICS Background checks yearly % change',

    xaxis_title="Year",

    yaxis_title='% Change')

fig3.show()
#Creating a new data frame that calculates the mean number of background checks for each state and territory

permit_by_state = firearm_data_w_pop.groupby("state")["totals"].mean()

#Merging the newly created data frame with the state data

firearm_data_w_pop = firearm_data_w_pop.reset_index().merge(permit_by_state, on='state', suffixes = ('_month', '_average')).set_index('date')

#Rounding the mean column to whole number

firearm_data_w_pop = firearm_data_w_pop.round()

#Converting the mean column from a float to an integer

firearm_data_w_pop['totals_average'] = firearm_data_w_pop['totals_average'].astype(int)

#Calculating the number of background checks per 10000 inhabitants in each state

firearm_data_w_pop['permits_per_10k'] =  firearm_data_w_pop['totals_average'] / firearm_data_w_pop['Pop'] * 10000

#Rounding the permits_per_10k column to whole number

firearm_data_w_pop = firearm_data_w_pop.round()

#Converting the permits_per_10k column from a float to an integer

firearm_data_w_pop['permits_per_10k'] = firearm_data_w_pop['permits_per_10k'].astype(int)

#Displaying the new data frame

firearm_data_w_pop.head()
#Dropping duplicates from each instance of the states to make for more streamlined analysis

firearm_data_w_pop_ind = firearm_data_w_pop.drop_duplicates('state')

#Displaying new data frame

firearm_data_w_pop_ind.head()
#Sorting the data frame by permits per 10k

per_10k_sorted = firearm_data_w_pop_ind.sort_values('permits_per_10k', ascending=True)

#Displaying the top of the data frame showing the state or territory with the least permits per 10k

print(per_10k_sorted.head())

print('\n')

#Displaying the top of the data frame showing the state or territory with the most permits per 10k

print(per_10k_sorted.tail())
#Creating visualization to display the average number of background checks per state 

fig4 = px.bar(firearm_data_w_pop_ind, y='totals_average', x='state', color_discrete_sequence=px.colors.qualitative.Dark2)

fig4.update_layout(

title='Monthly background checks by state',

    xaxis_title="Year",

    yaxis_title='Average monthly background checks')

fig4.show()
#Creating visualization to display the number of background checks per 10k inhabitants

fig5 = px.bar(firearm_data_w_pop_ind, y='permits_per_10k', x='state')

fig5.update_layout(

title='Background checks per 10k inhabitants',

    xaxis_title="Year",

    yaxis_title='Background checks per 10k')

fig5.show()
#Creating visualization to display the number of background checks per 10k inhabitants

fig6 = px.bar(firearm_data_w_pop_ind, y='permits_per_10k', x='state', color = 'Pop')

fig6.update_layout(

title='Background checks per 10k inhabitants',

    xaxis_title="Year",

    yaxis_title='Background checks per 10k')

fig6.show()
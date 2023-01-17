import rkn_module_benford_law as rkn_benford



import sys

import csv

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import matplotlib.image as mpimg
path = "../input/inputbenfordcovid19/time_series_covid19_recovered_global.csv"



data = pd.read_csv(path, encoding='ISO-8859-1', delimiter=',' , low_memory=False)



data_mod = data.copy()



df = pd.DataFrame(data_mod)
# df to store the analysis

df_analysis = pd.DataFrame(columns=['var_name', 'var_NaN', 'var_not_NaN', "var_min", "var_max" , "var_mean" , 'var_type', 'var_categ'])
# Column names in dataSet

coluna_name_list = list(df.columns.values)
# This routine will store on df_analise the number os NaN each variable from dataset has. Alto the variable type and its categories.

for i in coluna_name_list:

    if(df[i].dtypes == "object"):

        lista=[i, df[i].isna().sum(), df[i].count(), "NA", "NA", "NA", df[i].dtypes, "numerical variable"]

        df_length = len(df_analysis)

        df_analysis.loc[df_length] = lista

    else:

        lista=[i, df[i].isna().sum(), df[i].count(), df[i].min(), df[i].max(), df[i].mean(), df[i].dtypes, "numerical variable"]

        df_length = len(df_analysis)

        df_analysis.loc[df_length] = lista
# Set var_name as index

df_analysis.set_index('var_name', inplace=True)
# For each non numerical variable assign its possible values

for i in coluna_name_list:

    if(df_analysis.loc[i, "var_type"] == "object"):

        df_analysis.loc[i, "var_categ"] = list(df[i].unique())

    else:

        pass
df_analysis.sort_values("var_NaN", ascending=False)
# Create a new column with country names. If a country has no provinces/state desaggregation, so it will show the term: Single Unity

df["Province/State"] = df["Province/State"].replace(np.NaN, "Single Unity")

df["Country_State"] = df["Country/Region"] + "_" + df["Province/State"]
# Remove unecessary columns

del df["Province/State"]

del df["Country/Region"]

del df["Lat"]

del df["Long"]
df
# Send columns to rows

df_melt = df.melt(id_vars=["Country_State"],

       var_name="Date",

       value_name="Recovered_Accumulated")
# Assign date type

df_melt['Date'] = pd.to_datetime(df_melt['Date'])
# Sort by country and date

df_melt = df_melt.sort_values(["Country_State", "Date"], ascending = (True, True))
# New column to receive the new cases for each day

df_melt["Recovered_New_Day"] = 0

# This function will assign to each country/day the number of new cases confirmed, based on the difference among accumulated cases of the actual and last day.

country_before = df_melt.iloc[0,0]



for row in range(1, df_melt.shape[0], 1):

    country_actual = df_melt.iloc[row,0]

    if(country_actual == country_before):

        df_melt.iloc[row,3] = df_melt.iloc[row,2] - df_melt.iloc[row-1,2]

    else:

        df_melt.iloc[row,3] = df_melt.iloc[row,2]

    country_before = country_actual
# Reset index: drop = False

df_melt.reset_index(inplace = True, drop = True)
# Create column to store de number of the week

df_melt['Date_week'] = pd.DatetimeIndex(df_melt['Date']).week
# New dataFrame grouped by country and week. The column value represents the number of new cases in each week per country.

df_agg_week = (df_melt.groupby(['Country_State', 'Date_week']).sum()).copy()
# Remove columns && rename columns

del df_agg_week["Recovered_Accumulated"]

df_agg_week = df_agg_week.rename(columns = {"Recovered_New_Day": "Recovered_New_Week"})
# Set index to columns: drop = False

df_agg_week.reset_index(inplace = True, drop = False)
# IMPORTANT: This is the data that will be used on our analysis. However, will keep on the code to create a more robust database

df_week = df_agg_week.copy()

del df_week["Date_week"]

df_week.to_excel("df_week_recovered.xlsx")
df_week
# Create key-column (Country_week) to join both dataFrames

df_melt["Country_week"] = df_melt["Country_State"] + "-" + df_melt["Date_week"].astype(str)

df_agg_week["Country_week"] = df_agg_week["Country_State"] + "-" + df_agg_week["Date_week"].astype(str)
# Reorder column

df_agg_week = df_agg_week[['Country_week', 'Recovered_New_Week']]

df_melt = df_melt[["Country_week", 'Country_State', 'Date', "Date_week", "Recovered_Accumulated", "Recovered_New_Day"]]
# Create the final dataFrame with number of accumulated cases, daily cases and weekly cases. Key-column: Country_week

df_merge = df_melt.merge(df_agg_week, on="Country_week")
# Remove key column

del df_merge["Country_week"]
# Show final data and send to excell

df_merge.to_excel("df_merge_recovered.xlsx")
# Note that we have few weeks of information per country. Insufficient for an analysis ungrouped per country.

df_merge
# Note that it is possible to have a negative new number of cases. Meaning that in such a week the government corrected the numbers informed in the previous week.

df_week.describe()
# Some deeper analysis

df_analysis_desc = (df_week.groupby(['Country_State']).describe()).copy()

df_analysis_desc
# Getting hints (1 for aggregated analysis)

rkn_benford.hints(df_week, 1)
# df_week: data set with the values to be analyzed

# 1: Aggregated analysis (Since the sample per country is very small, we are only interested to analyze frequencies of the entire data set as a whole.)

# 5: Number of rounds we will run the code in order to produced an averaged chi-squared value.

# 500: Sample size for the first digit analysis.

# 500: Sample size for the second digit analysis.

# 184: Sample size for the third digit analysis.

# 1: Number of graphs to produce with the best chi-sq values.

# 1: Number of graphs to produce with the worst chi-sq values. Same as the graph before.

table_app = rkn_benford.benford(df_week, 1, 5, 500, 500, 184, 1, 1, "output_recovered.xlsx", "")
# Order by city name

table_app[0].sort_values(by=['units'], inplace=True)

# Format table values

results_d1 = table_app[0].style.format({

    'N0': '{:,.2%}'.format, 'N1': '{:,.2%}'.format, 'N2': '{:,.2%}'.format, 'N3': '{:,.2%}'.format, 'N4': '{:,.2%}'.format, 'N5': '{:,.2%}'.format,

    'N6': '{:,.2%}'.format, 'N7': '{:,.2%}'.format, 'N8': '{:,.2%}'.format, 'N8': '{:,.2%}'.format, 'N9': '{:,.2%}'.format, 

    'chi_sq': '{:,.2f}'.format, 'chi_sq 10 rounds': '{:,.2f}'.format,

    })
fig = plt.figure(figsize=(8,6), dpi=250)



a = fig.add_subplot(1, 1, 1)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordcovid19/D1__Aggregated_recovered_table.png'))



plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
fig = plt.figure(figsize=(8,6), dpi=250)



a = fig.add_subplot(1, 1, 1)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordcovid19/D1__Aggregated_recovered.png'))



plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
# Order by city name

table_app[1].sort_values(by=['units'], inplace=True)

# Format table values

results_d2 = table_app[1].style.format({

    'N0': '{:,.2%}'.format, 'N1': '{:,.2%}'.format, 'N2': '{:,.2%}'.format, 'N3': '{:,.2%}'.format, 'N4': '{:,.2%}'.format, 'N5': '{:,.2%}'.format,

    'N6': '{:,.2%}'.format, 'N7': '{:,.2%}'.format, 'N8': '{:,.2%}'.format, 'N8': '{:,.2%}'.format, 'N9': '{:,.2%}'.format, 

    'chi_sq': '{:,.2f}'.format, 'chi_sq 10 rounds': '{:,.2f}'.format,

    })
fig = plt.figure(figsize=(8,6), dpi=250)



a = fig.add_subplot(1, 1, 1)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordcovid19/D2__Aggregated_recovered_table.png'))



plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
fig = plt.figure(figsize=(8,6), dpi=250)



a = fig.add_subplot(1, 1, 1)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordcovid19/D2__Aggregated_recovered.png'))



plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
# Order by city name

table_app[2].sort_values(by=['units'], inplace=True)

# Format table values

results_d3 = table_app[2].style.format({

    'N0': '{:,.2%}'.format, 'N1': '{:,.2%}'.format, 'N2': '{:,.2%}'.format, 'N3': '{:,.2%}'.format, 'N4': '{:,.2%}'.format, 'N5': '{:,.2%}'.format,

    'N6': '{:,.2%}'.format, 'N7': '{:,.2%}'.format, 'N8': '{:,.2%}'.format, 'N8': '{:,.2%}'.format, 'N9': '{:,.2%}'.format, 

    'chi_sq': '{:,.2f}'.format, 'chi_sq 10 rounds': '{:,.2f}'.format,

    })
fig = plt.figure(figsize=(8,6), dpi=250)



a = fig.add_subplot(1, 1, 1)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordcovid19/D3__Aggregated_recovered_table.png'))



plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
fig = plt.figure(figsize=(8,6), dpi=250)



a = fig.add_subplot(1, 1, 1)

imgplot = plt.imshow(mpimg.imread('../input/inputbenfordcovid19/D3__Aggregated_recovered.png'))



plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
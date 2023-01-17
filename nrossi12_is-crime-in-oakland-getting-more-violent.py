import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from pandas import Timestamp

from datetime import date

from dateutil.relativedelta import relativedelta

from sklearn.linear_model import LinearRegression
df_2011 = pd.read_csv('../input/oakland-crime-statistics-2011-to-2016/records-for-2011.csv', parse_dates=['Create Time', 'Closed Time'])

df_2012 = pd.read_csv('../input/oakland-crime-statistics-2011-to-2016/records-for-2012.csv', parse_dates=['Create Time', 'Closed Time'])

df_2013 = pd.read_csv('../input/oakland-crime-statistics-2011-to-2016/records-for-2013.csv', parse_dates=['Create Time', 'Closed Time'])

df_2014 = pd.read_csv('../input/oakland-crime-statistics-2011-to-2016/records-for-2014.csv', parse_dates=['Create Time', 'Closed Time'])

df_2015 = pd.read_csv('../input/oakland-crime-statistics-2011-to-2016/records-for-2015.csv', parse_dates=['Create Time', 'Closed Time'])

df_2016 = pd.read_csv('../input/oakland-crime-statistics-2011-to-2016/records-for-2016.csv', parse_dates=['Create Time', 'Closed Time'])
list_dfs = [df_2011, df_2012, df_2013, df_2014, df_2015, df_2016]
def shapes():

    x = 0

    for i in list_dfs:

        print(f"Shape of dataset for {x+2011} is {i.shape}")

        x+=1

shapes()
df_2011.head()
df_2012.head()
df_2013.head()
df_2014.head()
df_2015.head()
df_2016.head()
# Code to show count of priority crimes per year.

a = 0

for i in list_dfs:

    print(i[i['Priority']!=0].groupby(['Priority']).size().reset_index(name=str(f'Count in {a + 2011}')))

    a += 1

    print(' ')
# Bar charts for comparing priority type crimes

df = pd.DataFrame([

    [1, 36699, 41926, 43171, 42773, 42418, 24555],

    [2, 143314, 145504, 144859, 144707, 150162, 86272]

],

columns=['Priority']+[f'Count in {x}' for x in range(2011,2017)]

)



df.plot.bar(x='Priority', subplots=True, layout=(2,3), figsize=(15, 7))
pri1_2011 = 36699

pri2_2011 = 143314

total_2011 = pri1_2011 + pri2_2011

print(f"Priority 1 crimes amounted to {round((pri1_2011/total_2011)*100, 3)}%, priority 2 crimes amounted to {round((pri2_2011/total_2011)*100, 3)}% in 2011.")

print("-----------------------------------------------------------------------------------------------------------------------------------------")

pri1_2012 = 41926

pri2_2012 = 145504

total_2012 = pri1_2012 + pri2_2012

print(f"Priority 1 crimes amounted to {round((pri1_2012/total_2012)*100, 3)}%, priority 2 crimes amounted to {round((pri2_2012/total_2012)*100, 3)}% in 2012.")

print("-----------------------------------------------------------------------------------------------------------------------------------------")

pri1_2013 = 43171

pri2_2013 = 144859

total_2013 = pri1_2013 + pri2_2013

print(f"Priority 1 crimes amounted to {round((pri1_2013/total_2013)*100, 3)}%, priority 2 crimes amounted to {round((pri2_2013/total_2013)*100, 3)}% in 2013.")

print("-----------------------------------------------------------------------------------------------------------------------------------------")

pri1_2014 = 42773

pri2_2014 = 144707

total_2014 = pri1_2014 + pri2_2014

print(f"Priority 1 crimes amounted to {round((pri1_2014/total_2014)*100, 3)}% priority 2 crimes amounted to {round((pri2_2014/total_2014)*100, 3)}% in 2014.")

print("-----------------------------------------------------------------------------------------------------------------------------------------")

pri1_2015 = 42418

pri2_2015 = 150162

total_2015 = pri1_2015 + pri2_2015

print(f"Priority 1 crimes amounted to {round((pri1_2015/total_2015)*100, 3)}%, priority 2 crimes amounted to {round((pri2_2015/total_2015)*100, 3)}% in 2015.")

print("-----------------------------------------------------------------------------------------------------------------------------------------")

pri1_2016 = 24555

pri2_2016 = 86272

total_2016 = pri1_2016 + pri2_2016

print(f"Priority 1 crimes amounted to {round((pri1_2016/total_2016)*100, 3)}% and priority 2 crimes amounted to {round((pri2_2016/total_2016)*100, 3)}%, for the first half of 2016.")

print("-----------------------------------------------------------------------------------------------------------------------------------------")
# Mean Priority count per Area/Location/Beat

def areaid_groupby():

    for i in list_dfs:

        print(i[i['Priority']!=0].groupby(['Area Id', 'Priority']).size())

        print(' ')

areaid_groupby()
fig, axes= plt.subplots(2, 3)

for i, d in enumerate(list_dfs):

    ax = axes.flatten()[i]

    dplot = d[['Area Id', 'Priority']].pivot_table(index='Area Id', columns=['Priority'], aggfunc=len)

    dplot = (dplot.assign(total=lambda x: x.sum(axis=1))

                  .sort_values('total', ascending=False)

                  .head(10)

                  .drop('total', axis=1))

    dplot.plot.bar(ax=ax, figsize=(15, 7), stacked=True)

    ax.set_title(f"Plot of Priority 1 and 2 crimes within Area Id for {i+2011}")

    plt.tight_layout()
# Value count for beats displayed by priority 

for i in list_dfs:

    print(i[i['Priority']!=0].groupby(['Beat', 'Priority']).size())

    print(' ')
fig, axes = plt.subplots(2, 3)

for i, d in enumerate(list_dfs):

    ax = axes.flatten()[i]

    dplot = d[['Beat', 'Priority']].pivot_table(index='Beat', columns=['Priority'], aggfunc=len)

    dplot = (dplot.assign(total=lambda x: x.sum(axis=1))

                  .sort_values('total', ascending=False)

                  .head(10)

                  .drop('total', axis=1))

    dplot.plot.bar(ax=ax, figsize=(15, 7), stacked=True)

    ax.set_title(f"Top 10 Beats for {i+ 2011}")

    plt.tight_layout()
# Top 20 most popular crimes across the data sets

df1 = df_2011['Incident Type Description'].value_counts()[:10]

df2 = df_2012['Incident Type Description'].value_counts()[:10]

df3 = df_2013['Incident Type Description'].value_counts()[:10]

df4 = df_2014['Incident Type Description'].value_counts()[:10]

df5 = df_2015['Incident Type Description'].value_counts()[:10]

df6 = df_2016['Incident Type Description'].value_counts()[:10]

list_df = [df1, df2, df3, df4, df5, df6]

fig, axes = plt.subplots(2, 3)

for d, i in zip(list_df, range(6)):

    ax=axes.ravel()[i];

    ax.set_title(f"Top 20 crimes in {i+2011}")

    d.plot.barh(ax=ax, figsize=(15, 7))

    plt.tight_layout()
fig, axes = plt.subplots(2, 3)

for i, d in enumerate(list_dfs):

    ax = axes.flatten()[i]

    dplot = d[['Incident Type Id', 'Priority']].pivot_table(index='Incident Type Id', columns='Priority',aggfunc=len)

    dplot = (dplot.assign(total=lambda x: x.sum(axis=1))

                  .sort_values('total', ascending=False)

                  .head(10)

                  .drop('total', axis=1))

    dplot.plot.barh(ax=ax, figsize=(15, 7), stacked=True)

    ax.set_title(f"Plot of Top 10 Incidents in {i+2011}")

    plt.tight_layout()
# Total amount of pripority crimes per month

pri_count_list = [df_2011.groupby(['Priority', df_2011['Create Time'].dt.to_period('m')]).Priority.count(),

                  df_2012.groupby(['Priority', df_2012['Create Time'].dt.to_period('m')]).Priority.count(),

                  df_2013.groupby(['Priority', df_2013['Create Time'].dt.to_period('m')]).Priority.count(),

                  df_2014.groupby(['Priority', df_2014['Create Time'].dt.to_period('m')]).Priority.count(),

                  df_2015.groupby(['Priority', df_2015['Create Time'].dt.to_period('m')]).Priority.count(),

                  df_2016.groupby(['Priority', df_2016['Create Time'].dt.to_period('m')]).Priority.count()]

fig, axes = plt.subplots(2, 3)

for d, ax in zip(pri_count_list, axes.ravel()):

    plot_df1 = d.unstack('Priority').loc[:, 1]

    plot_df2 = d.unstack('Priority').loc[:, 2]

    plot_df1.index = pd.PeriodIndex(plot_df1.index.tolist(), freq='m')

    plot_df2.index = pd.PeriodIndex(plot_df2.index.tolist(), freq='m')

    plt.suptitle('Visualisation of priorities by the year')

    plot_df1.plot(ax=ax, legend=True, figsize=(15, 7))

    plot_df2.plot(ax=ax, legend=True, figsize=(15, 7))
count = 2011

x = []

for i in list_dfs:

    i['Difference in hours'] = i['Closed Time'] - i['Create Time']

    i['Difference in hours'] = i['Difference in hours']/np.timedelta64(1, 'h')

    mean_hours = round(i['Difference in hours'].mean(), 3)

    x.append(mean_hours)

    print(f"Difference in hours for {count} is {mean_hours} with a reported {i.shape[0]} crimes.")

    count += 1

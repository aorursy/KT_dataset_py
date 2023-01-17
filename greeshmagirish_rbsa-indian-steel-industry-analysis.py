import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='whitegrid')
import_df = pd.read_csv("../input/Finished_Steel_Import_Categorywise_1_1.csv")

export_df = pd.read_csv("../input/Finished_Steel_Export_Categorywise_1.csv")
import_df.info()
export_df.info()
import_df.head(2)
export_df.head(2)
print(import_df.Category.unique())

print(export_df.Category.unique())

print(import_df['Sub-Category'].unique())

print(export_df['Sub-Category'].unique())

import_df = import_df[(import_df.Category != 'Total Import of Finished Steel')]

export_df = export_df[(export_df.Category != 'Total Export of Finished Steel')]

import_df = import_df[(import_df['Sub-Category'] != 'Total - Non-Alloy Finished Steel') & (import_df['Sub-Category'] != 'Total - Alloy/Stainless Finished Steel')]

export_df = export_df[(export_df['Sub-Category'] != 'Total - Non-Alloy Finished Steel') & (export_df['Sub-Category'] != 'Total - Alloy/Stainless Finished Steel')]

print(import_df.Category.unique())

print(export_df.Category.unique())

print(import_df['Sub-Category'].unique())

print(export_df['Sub-Category'].unique())
cat_import_df = import_df.groupby(by=['Category'])['2013-14','2014-15','2015-16','2016-17','2017-18'].sum().reset_index().sort_values(by=['2013-14','2014-15','2015-16','2016-17','2017-18'], ascending=False)

cat_import_df['Total'] = cat_import_df.apply(lambda row: row['2013-14'] + row['2014-15'] + row['2015-16'] + row['2016-17'] + row['2017-18'], axis = 1) 



cat_export_df = export_df.groupby(by=['Category'])['2013-14','2014-15','2015-16','2016-17','2017-18'].sum().reset_index().sort_values(by=['2013-14','2014-15','2015-16','2016-17','2017-18'], ascending=False)

cat_export_df['Total'] = cat_export_df.apply(lambda row: row['2013-14'] + row['2014-15'] + row['2015-16'] + row['2016-17'] + row['2017-18'], axis = 1) 





fig,ax = plt.subplots(2, 1, figsize=(20,10), sharey=True)

plt.subplots_adjust(hspace=0.3)

sns.barplot(cat_import_df['Category'],cat_import_df['Total'],ax=ax[0])

ax[0].set_title("Import")

ax[0].set_ylabel('Total')

sns.barplot(cat_export_df['Category'],cat_export_df['Total'],ax=ax[1])

ax[1].set_title("Export")

ax[1].set_ylabel('Total')

fig.suptitle('Total Import/Export from 2013-2018',fontsize=18)
years = ['2013-14','2014-15','2015-16','2016-17','2017-18']



cat_import_df = import_df.groupby(by=['Category'])['2013-14','2014-15','2015-16','2016-17','2017-18'].sum().reset_index().sort_values(by=['2013-14','2014-15','2015-16','2016-17','2017-18'], ascending=False)



fig,ax = plt.subplots(1,1, figsize=(20,10))

for i,year in enumerate(years):

    sns.lineplot(x=cat_import_df['Category'],y=cat_import_df[year], label=years[i])

    plt.ylabel('Quantity in Tonnes')

    plt.title('Import of Finished Steel', fontsize=15)





cat_export_df = export_df.groupby(by=['Category'])['2013-14','2014-15','2015-16','2016-17','2017-18'].sum().reset_index().sort_values(by=['2013-14','2014-15','2015-16','2016-17','2017-18'], ascending=False)





fig,ax = plt.subplots(1,1, figsize=(20,10))

for i,year in enumerate(years):

    sns.lineplot(x=cat_export_df['Category'],y=cat_export_df[year], label=years[i])

    plt.ylabel('Quantity in Tonnes')

    plt.title('Export of Finished Steel', fontsize=15)
df1 = import_df[(import_df.Category == 'Non-Alloy Finished Steel')]

df2 = export_df[(export_df.Category == 'Non-Alloy Finished Steel')]

sub_import_df = df1.groupby(by=['Sub-Category'])['2013-14','2014-15','2015-16','2016-17','2017-18'].sum().reset_index().sort_values(by=['2013-14','2014-15','2015-16','2016-17','2017-18'], ascending=False)

sub_import_df['Total'] = cat_import_df.apply(lambda row: row['2013-14'] + row['2014-15'] + row['2015-16'] + row['2016-17'] + row['2017-18'], axis = 1) 



sub_export_df = df2.groupby(by=['Sub-Category'])['2013-14','2014-15','2015-16','2016-17','2017-18'].sum().reset_index().sort_values(by=['2013-14','2014-15','2015-16','2016-17','2017-18'], ascending=False)

sub_export_df['Total'] = sub_export_df.apply(lambda row: row['2013-14'] + row['2014-15'] + row['2015-16'] + row['2016-17'] + row['2017-18'], axis = 1) 



fig,ax = plt.subplots(2, 1, figsize=(20,10), sharey=True)

plt.subplots_adjust(hspace=0.3)

sns.barplot(sub_import_df['Sub-Category'],sub_import_df['Total'],ax=ax[0])

ax[0].set_title("Import")

ax[0].set_ylabel('Total')

sns.barplot(sub_export_df['Sub-Category'],sub_export_df['Total'],ax=ax[1])

ax[1].set_title("Export")

ax[1].set_ylabel('Total')

fig.suptitle('Total Non Alloy Finished Steel Import/Export from 2013-2018')
years = ['2013-14','2014-15','2015-16','2016-17','2017-18']



sub_df = df1[(df1['Sub-Category'] == 'Flat Products of Non-Alloy Finished Steel')]

sub_df = sub_df.drop(['Category', 'Sub-Category'], axis=1)

sub_df = sub_df.rename(columns = {'''Item - (Import Quantity in '000 Tonnes)''': 'Items'})

sub_df = sub_df[(sub_df.Items != 'Total - Flat Products of Non-Alloy Finished Steel')]

sub_df = sub_df.fillna(0)



fig,ax = plt.subplots(1,1, figsize=(20,10))

for i,year in enumerate(years):

    sns.lineplot(x=sub_df['Items'],y=sub_df[year], label=years[i])

    plt.ylabel('Quantity')

    plt.xticks(rotation=90)

    plt.title('Import: Flat Products - Non Alloy Finished Steel', fontsize=15)
sub_df = df2[(df2['Sub-Category'] == 'Flat Products of Non-Alloy Finished Steel')]

sub_df = sub_df.drop(['Category', 'Sub-Category'], axis=1)

sub_df = sub_df.rename(columns = {'''Item - (Export Quantity in '000 Tonnes)''': 'Items'})

sub_df = sub_df[(sub_df.Items != 'Total - Flat Products of Non-Alloy Finished Steel')]

sub_df = sub_df.fillna(0)





fig,ax = plt.subplots(1,1, figsize=(20,10))

for i,year in enumerate(years):

    sns.lineplot(x=sub_df['Items'],y=sub_df[year], label=years[i])

    plt.ylabel('Quantity')

    plt.xticks(rotation=90)

    plt.title('Export: Flat Products - Non Alloy Finished Steel', fontsize=15)
sub_df = df1[(df1['Sub-Category'] == 'Non-Flat Products of Non-Alloy Finished Steel')]

sub_df = sub_df.drop(['Category', 'Sub-Category'], axis=1)

sub_df = sub_df.rename(columns = {'''Item - (Import Quantity in '000 Tonnes)''': 'Items'})

sub_df = sub_df[(sub_df.Items != 'Total - Non-Flat Products of Non-Alloy Finished Steel')]

sub_df = sub_df.fillna(0)



fig,ax = plt.subplots(1,1, figsize=(20,10))

for i,year in enumerate(years):

    sns.lineplot(x=sub_df['Items'],y=sub_df[year], label=years[i])

    plt.ylabel('Quantity')

    plt.title('Import: Non-Flat Products - Non Alloy Finished Steel', fontsize=15)
sub_df = df2[(df2['Sub-Category'] == 'Non-Flat Products of Non-Alloy Finished Steel')]

sub_df = sub_df.drop(['Category', 'Sub-Category'], axis=1)

sub_df = sub_df.rename(columns = {'''Item - (Export Quantity in '000 Tonnes)''': 'Items'})

sub_df = sub_df[(sub_df.Items != 'Total - Non-Flat Products of Non-Alloy Finished Steel')]

sub_df = sub_df.fillna(0)





fig,ax = plt.subplots(1,1, figsize=(20,10))

for i,year in enumerate(years):

    sns.lineplot(x=sub_df['Items'],y=sub_df[year], label=years[i])

    plt.ylabel('Quantity')

    plt.title('Export: Non-Flat Products - Non Alloy Finished Steel', fontsize=15)
df1 = import_df[(import_df.Category == 'Alloy/Stainless Finished Steel')]

df2 = export_df[(export_df.Category == 'Alloy/Stainless Finished Steel')]

sub_import_df = df1.groupby(by=['Sub-Category'])['2013-14','2014-15','2015-16','2016-17','2017-18'].sum().reset_index().sort_values(by=['2013-14','2014-15','2015-16','2016-17','2017-18'], ascending=False)

sub_import_df['Total'] = cat_import_df.apply(lambda row: row['2013-14'] + row['2014-15'] + row['2015-16'] + row['2016-17'] + row['2017-18'], axis = 1) 



sub_export_df = df2.groupby(by=['Sub-Category'])['2013-14','2014-15','2015-16','2016-17','2017-18'].sum().reset_index().sort_values(by=['2013-14','2014-15','2015-16','2016-17','2017-18'], ascending=False)

sub_export_df['Total'] = sub_export_df.apply(lambda row: row['2013-14'] + row['2014-15'] + row['2015-16'] + row['2016-17'] + row['2017-18'], axis = 1) 



fig,ax = plt.subplots(2, 1, figsize=(20,10), sharey=True)

plt.subplots_adjust(hspace=0.3)

sns.barplot(sub_import_df['Sub-Category'],sub_import_df['Total'],ax=ax[0])

ax[0].set_title("Import")

ax[0].set_ylabel('Total')

sns.barplot(sub_export_df['Sub-Category'],sub_export_df['Total'],ax=ax[1])

ax[1].set_title("Export")

ax[1].set_ylabel('Total')

fig.suptitle('Total Alloy/Stainless Finished Steel Import/Export from 2013-2018')
years = ['2013-14','2014-15','2015-16','2016-17','2017-18']



sub_df = df1[(df1['Sub-Category'] == 'Flat Products of Alloy/Stainless Finished Steel')]

sub_df = sub_df.drop(['Category', '''Item - (Import Quantity in '000 Tonnes)'''], axis=1)

sub_df = sub_df.T.reset_index()

sub_df = sub_df.rename(columns = {'index': 'Years', 18: 'Quantity'})

sub_df = sub_df.drop(sub_df.index[0])

sub_df



fig,ax = plt.subplots(figsize=(20,10))

sns.barplot(x=sub_df['Years'],y=sub_df['Quantity'], label=sub_df['Years'])

plt.ylabel('Quantity')

plt.title('Import: Flat Products of Alloy/Stainless Finished Steel', fontsize=15)
years = ['2013-14','2014-15','2015-16','2016-17','2017-18']



sub_df = df2[(df2['Sub-Category'] == 'Flat Products of Alloy/Stainless Finished Steel')]

sub_df = sub_df.drop(['Category', '''Item - (Export Quantity in '000 Tonnes)'''], axis=1)

sub_df = sub_df.T.reset_index()

sub_df = sub_df.rename(columns = {'index': 'Years', 17: 'Quantity'})

sub_df = sub_df.drop(sub_df.index[0])

sub_df



fig,ax = plt.subplots(figsize=(20,10))

sns.barplot(x=sub_df['Years'],y=sub_df['Quantity'], label=sub_df['Years'])

plt.ylabel('Quantity')

plt.title('Export: Flat Products of Alloy/Stainless Finished Steel', fontsize=15)
sub_df = df1[(df1['Sub-Category'] == 'Non-Flat Products of Alloy/Stainless Finished Steel')]

sub_df = sub_df.drop(['Category', '''Item - (Import Quantity in '000 Tonnes)'''], axis=1)

sub_df = sub_df.T.reset_index()

sub_df = sub_df.rename(columns = {'index': 'Years', 17: 'Quantity'})

sub_df = sub_df.drop(sub_df.index[0])

sub_df



fig,ax = plt.subplots(figsize=(20,10))

sns.barplot(x=sub_df['Years'],y=sub_df['Quantity'], label=sub_df['Years'])

plt.ylabel('Quantity')

plt.title('Import: Non- Flat Products of Alloy/Stainless Finished Steel', fontsize=15)
sub_df = df2[(df2['Sub-Category'] == 'Non-Flat Products of Alloy/Stainless Fnished Steel')]

sub_df = sub_df.drop(['Category', '''Item - (Export Quantity in '000 Tonnes)'''], axis=1)

sub_df = sub_df.T.reset_index()

sub_df = sub_df.rename(columns = {'index': 'Years', 16: 'Quantity'})

sub_df = sub_df.drop(sub_df.index[0])

sub_df



fig,ax = plt.subplots(figsize=(20,10))

sns.barplot(x=sub_df['Years'],y=sub_df['Quantity'], label=sub_df['Years'])

plt.ylabel('Quantity')

plt.title('Export: Non-Flat Products of Alloy/Stainless Finished Steel', fontsize=15)
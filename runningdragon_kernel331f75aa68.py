# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Print export and import data from csv to pandas datastructure

raw_export_data=pd.read_csv('/kaggle/input/india-trade-data/2018-2010_export.csv')

raw_import_data=pd.read_csv('/kaggle/input/india-trade-data/2018-2010_import.csv')



#Print Column titles

print(raw_export_data.columns)

print(raw_import_data.columns)
#Matrix size of export and import tables

print(raw_export_data.shape)

print(raw_import_data.shape)

print(raw_export_data.head(10))

print(raw_import_data.head(10))
#Number of Unique value in HSCode

export_hscode=raw_export_data["HSCode"]

print(export_hscode.unique().size)

import_hscode=raw_import_data["HSCode"]

print(import_hscode.unique().size)



#Number of Unique value in HSCode and Commodity

export_hscode_comm=raw_export_data[['HSCode','Commodity']]

unique_export_hscode_comm = export_hscode_comm.drop_duplicates()

print(unique_export_hscode_comm.shape)

import_hscode_comm=raw_import_data[['HSCode','Commodity']]

unique_import_hscode_comm = import_hscode_comm.drop_duplicates()

print(unique_import_hscode_comm.shape)



#Same number of unique value confirms 1 to 1 mapping. We can used only ID for any future data analysis

export_data=raw_export_data.drop(columns=['Commodity'])

import_data=raw_import_data.drop(columns=['Commodity'])

print(export_data.shape)

print(import_data.shape)
#Number of countries in  trade for export and import

export_country=export_data['country']

print(export_country.unique().size)

import_country=import_data['country']

print(import_country.unique().size)
#Find columns with missing or NAN values

print(export_data.isna().any())

print(import_data.isna().any())



#Clean NAN valuews and replace with 0

export_data=export_data.fillna(0)

import_data=import_data.fillna(0)



print(export_data.isna().any())

print(import_data.isna().any())



import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(15,7))

ax = plt.gca()

fig2, ax2 = plt.subplots(figsize=(15,7))

fig3, ax3 = plt.subplots(figsize=(15,7))

fig4, ax4 = plt.subplots(figsize=(15,7))





#Commodity by year

export_data.groupby(['year','HSCode']).sum()['value'].unstack().plot(ax=ax)

import_data.groupby(['year','HSCode']).sum()['value'].unstack().plot(ax=ax2)

#Country by year

export_data.groupby(['year','country']).sum()['value'].unstack().plot(ax=ax3)

import_data.groupby(['year','country']).sum()['value'].unstack().plot(ax=ax4)



#Top 5 Export and Import countries

largest_export_countries=export_data.groupby(['country']).sum()['value'].nlargest()

largest_import_countries=import_data.groupby(['country']).sum()['value'].nlargest()

print("Top 5 export countries")

print(largest_export_countries.index)

print("Top 5 import countries")

print(largest_import_countries.index)



#Top 5 Export and Import commodity by HSCODE

largest_export_hscode=export_data.groupby(['HSCode']).sum()['value'].nlargest()

largest_import_hscode=import_data.groupby(['HSCode']).sum()['value'].nlargest()

print(largest_export_hscode.index)

for exp_hscode in largest_export_hscode.index:

    print(unique_export_hscode_comm.loc[unique_export_hscode_comm['HSCode']==exp_hscode])

print(largest_import_hscode.index)

for imp_hscode in largest_import_hscode.index:

    print(unique_import_hscode_comm.loc[unique_import_hscode_comm['HSCode']==imp_hscode])
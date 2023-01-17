# Supress Warnings

import warnings

warnings.filterwarnings('ignore')
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os, sys



pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)

plt.rcParams["figure.figsize"] = [12,8]
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
india_export = pd.read_csv(r"/kaggle/input/india-trade-data/2018-2010_export.csv")

india_export.head()
india_import = pd.read_csv(r"/kaggle/input/india-trade-data/2018-2010_import.csv")

india_import.head()
# export data

Year_wise_export = india_export.groupby("year").sum()

Year_wise_export = Year_wise_export.drop("HSCode", axis=1).reset_index()



# import data

Year_wise_import = india_import.groupby("year").sum()

Year_wise_import = Year_wise_import.drop("HSCode", axis=1).reset_index()



Year_wise_trade = pd.DataFrame(Year_wise_export)

Year_wise_trade = Year_wise_trade.rename(index=str, columns={"value": "Export Value", "year" : "Year" })

import_value = Year_wise_import.value

Year_wise_trade["Import Value"] = list(import_value)

Year_wise_trade
plt.plot(Year_wise_trade.Year, Year_wise_trade["Export Value"])

plt.plot(Year_wise_trade.Year, Year_wise_trade["Import Value"])



plt.legend(['y = Export Value', 'y = Import Value'], loc='upper left')

plt.show()
import_commodity_wise = india_import.groupby(["Commodity"]).sum().drop(["HSCode", "year"], axis=1).reset_index()

import_commodity_wise = import_commodity_wise.sort_values(by ='value' , ascending=False)
india_import_pivot = pd.pivot_table(india_import,values='value',index = 'Commodity',columns='year').reset_index()
india_import_pivot["Total"] = import_commodity_wise.value

india_import_pivot = india_import_pivot.sort_values(by ='Total' , ascending=False)
top_five_commodity = india_import_pivot.sort_values(by ='Total' , ascending=False).head(5).reset_index()

top_five_commodity = top_five_commodity.drop(["index","Total"], axis=1)
top_five_commodity.set_index("Commodity")
top_five_commodity = top_five_commodity.transpose().reset_index()
new_header = top_five_commodity.iloc[0] #grab the first row for the header

top_five_commodity = top_five_commodity[1:] #take the data less the header row

top_five_commodity.columns = new_header #set the header row as the df header
top_five_commodity
top_five_commodity = top_five_commodity.rename(index=str, columns={"Commodity": "Year"})

top_five_commodity = top_five_commodity.set_index("Year")

top_five_commodity
india_import_with_country = india_import.groupby("country").sum()

india_import_with_country = india_import_with_country.drop(["year","HSCode"], axis = 1)

india_import_with_country = india_import_with_country.sort_values(by ='value' , ascending=False)

india_import_with_country_top_100 = india_import_with_country.head(100).reset_index()
plt.figure(figsize=(20,10))

country_wise_plot = sns.barplot(x = "country", y= "value", data=india_import_with_country_top_100)

country_wise_plot.set_xticklabels(country_wise_plot.get_xticklabels(), rotation=90)

plt.show()
india_export_with_country = india_export.groupby("country").sum()

india_export_with_country = india_export_with_country.drop(["year", "HSCode"], axis=1).sort_values(by = "value", ascending = False)

india_export_with_country_top_100 = india_export_with_country.head(100).reset_index()

india_export_with_country_top_100.head(10)
plt.figure(figsize=(20,10))

country_wise_plot_export = sns.barplot(x = "country", y= "value", data=india_export_with_country_top_100)

country_wise_plot_export.set_xticklabels(country_wise_plot_export.get_xticklabels(), rotation=90)

plt.show()
trade_relation_with_country = india_export_with_country.merge(india_import_with_country, on="country", how='inner').reset_index()
trade_relation_with_country = trade_relation_with_country.rename(index=str, columns={"value_x": "Export Value", "value_y": "Import Value"})

trade_relation_with_country_top = trade_relation_with_country.head(50)

trade_relation_with_country_top.head()
import squarify 



country = list(trade_relation_with_country_top.country)

Import_value = list(trade_relation_with_country_top["Import Value"])

Export_value = list(trade_relation_with_country_top["Export Value"])

Import_modified = []

Export_modified = []

title = []

for (x,y) in zip(Import_value,Export_value):

    x = round(x,0)

    y = round(y,0)

    Import_modified.append(x)

    Export_modified.append(y)

    

for (v, n, m) in zip(country, Import_modified, Export_modified):

    title.append(v+'\n Import = '+str(n)+'\n Export = '+str(m))

    

plt.rcParams['figure.figsize'] = (20.0, 20.0)

squarify.plot(sizes=Import_modified, label=title, alpha=0.6 )

plt.figure()

plt.axis('OFF')

plt.show()
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import pyplot

import seaborn as sns

sns.set(style="whitegrid")



avocado = pd.read_csv("/kaggle/input/avocado-prices/avocado.csv", index_col=0)



# sorting data frame by Date

avocado.sort_values("Date", axis=0, ascending=True, inplace=True, na_position='last')

avocado.head(10)
print (avocado.isnull().sum())
lowest_price_year = avocado.groupby("year")["AveragePrice"].min().reset_index()

average_price_year = avocado.groupby("year")["AveragePrice"].mean().reset_index()

highest_price_year = avocado.groupby("year")["AveragePrice"].max().reset_index()



price_table = pd.DataFrame({"Year":lowest_price_year["year"],

                            "Lowest average":lowest_price_year["AveragePrice"],

                            "Overall average":average_price_year["AveragePrice"],

                            "Highest average":highest_price_year["AveragePrice"]})

price_table.head()
plt.figure(figsize=(20,10))

sns.boxplot(x=avocado["year"], y=avocado["AveragePrice"], palette="rainbow")

plt.xticks(rotation=90)

plt.xlabel('Year')

plt.tick_params(labelsize = 15)

plt.ylabel('Average Price')

plt.title('Average Price of Avocado According to Year')
#Sorted (low > high) overall average price per region

grouped = avocado.groupby("region")["AveragePrice"].mean().reset_index()

sorted_regio_mean = grouped.sort_values("AveragePrice", ascending=True)

sorted_regio_mean.head()
plt.figure(figsize=(20,10))

sns.barplot(x=sorted_regio_mean["region"], y=sorted_regio_mean["AveragePrice"], palette="icefire_r")

plt.xticks(rotation=90)

plt.xlabel('Region')

plt.tick_params(labelsize = 15)

plt.ylabel('Average Price')

plt.title('Average Price of Avocado According to Region')
sns.boxplot(y="AveragePrice", x="type", data=avocado, palette = 'Set3')
sns.barplot(y="Total Volume", x="type", data=avocado, palette = 'Set3')
conventional = avocado[avocado.type=="conventional"]

organic = avocado[avocado.type=="organic"]

groupBy1_price = conventional.groupby('Date').mean()

groupBy2_price = organic.groupby('Date').mean()



df = pd.DataFrame({

    'Conventional':groupBy1_price.AveragePrice,

    'Organic': groupBy2_price.AveragePrice

    }, index=groupBy1_price.AveragePrice.index)

lines = df.plot.line(figsize=(20,10))

plt.tick_params(labelsize = 15)

plt.xlabel('Year')

plt.ylabel('Average Price')

plt.title('Average Price of Avocado According to Type')
sns.catplot('AveragePrice','region',data=avocado,

                   hue='year',

                   palette='Set3', kind="box",

                height=20, aspect=0.6, width=1.1

              )
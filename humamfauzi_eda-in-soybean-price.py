# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import pprint
df_train = pd.read_csv("../input/" + os.listdir("../input")[0])
df_train.Date = pd.to_datetime(df_train.Date)
df_train.set_index("Date", inplace=True)
# Any results you write to the current directory are saved as output.
us_regions = ["soux_H", 
              "soux_L", 
              "indianap_H", 
              "Indianap_L", 
              "memphis_H", 
              "memphis_L"]

commodity_price = ["Gold", "USD", "Oil"]

soy_OCHLV = ['Soy_Bean_high', 'Soy_Bean_low', 'Soy_Bean_settle',
       'Soy_Bean_volume', 'Soy_Bean_openint', 'Soy_Meal_high', 'Soy_Meal_low',
       'Soy_Meal_settle', 'Soy_Meal_volume', 'Soy_Meal_openint',
       'Soy_Oil_high', 'Soy_Oil_low', 'Soy_Oil_settle', 'Soy_Oil_volume',
       'Soy_Oil_openint']

production_origin = ['US_Area', 'US_Production', 'Brazil_Area',
       'Brazil_Production', 'Argentina_Area', 'Argentina_Production',
       'China_Area', 'China_Production', 'India_Area', 'India_Production',
       'Paraguay_Area', 'Paraguay_Production', 'Canada_Area',
       'Canada_Production', 'RussianF_Area', 'RussianF_Production',
       'CentAmer_Area', 'CentAmer_Production', 'Bolivia_Area',
       'Bolivia_Production', 'Africa_Area', 'Africa_Production']
plt.figure(figsize=(20,6))
sns.distplot(df_train["soux_H"], label="SOUX FALL")
sns.distplot(df_train["indianap_H"], label="INDIANAPOLIS")
sns.distplot(df_train["memphis_H"], label= "MEMPHIS")
plt.title("HIGH TEMPERATURE DISTRIBUTION")
plt.xlabel("Temperature")
plt.ylabel("Probability")
plt.legend()
plt.grid()

plt.figure(figsize=(20,6))
sns.distplot(df_train["soux_L"], label="SOUX FALL")
sns.distplot(df_train["indianap_L"], label="INDIANAPOLIS")
sns.distplot(df_train["memphis_L"], label= "MEMPHIS")
plt.title("LOW TEMPERATURE DISTRIBUTION")
plt.xlabel("Temperature")
plt.ylabel("Probability")
plt.legend()
plt.grid()
plt.figure(figsize=(20,6))
plt.title("HIGH TEMPERATURE PLOT IN ONE YEAR")
plt.plot(df_train['soux_H'].loc['1963-01-01':])
plt.plot(df_train['indianap_H'].loc['1963-01-01':])
plt.plot(df_train['memphis_H'].loc['1963-01-01':])
plt.legend(["SOUX FALL", "INDIANAPOLIS", "MEMPHIS"])
plt.grid()

plt.figure(figsize=(20,6))
plt.title("LOW TEMPERATURE PLOT IN ONE YEAR")
plt.plot(df_train['soux_L'].loc['1963-01-01':])
plt.plot(df_train['indianap_L'].loc['1963-01-01':])
plt.plot(df_train['memphis_L'].loc['1963-01-01':])
plt.legend(["SOUX FALL", "INDIANAPOLIS", "MEMPHIS"])
plt.grid()
soux_mid_temp = ((df_train['soux_H'] - df_train['soux_L']) / 2) + df_train['soux_L']
indianap_mid_temp = ((df_train['indianap_H'] - df_train['indianap_L']) / 2) + df_train['indianap_L']
soux_mid_temp = ((df_train['memphis_H'] - df_train['memphis_L']) / 2) + df_train['memphis_L']

avg_temp = (soux_mid_temp + indianap_mid_temp + soux_mid_temp) / 3
yearly_temp_avg = [avg_temp.loc[str(i)].mean() for i in np.unique(df_train.index.year)]
US_soybean_prod = [df_train["US_Production"].loc[str(i)].mean() for i in np.unique(df_train.index.year)]

# deleting 1961 because the temperature only recorded in winter
del yearly_temp_avg[0]
del US_soybean_prod[0]

plt.figure(figsize=(14,14))
plt.title("TEMPERATURE AND SOYBEAN PRODUCTION")
plt.scatter(yearly_temp_avg, US_soybean_prod)
plt.plot(yearly_temp_avg, US_soybean_prod, alpha=0.3)
plt.xlabel("Temperature Averange")
plt.ylabel("Soybean Production")
plt.grid()

fig, ax1 = plt.subplots(figsize=(15,6))
ax1.set_title("TEMPERATURE AND SOYBEAN PRODUCTION")
ax1.plot(np.unique(df_train.index.year)[1:], yearly_temp_avg, '-r')
ax1.set_xlabel('Year')
ax1.set_ylabel('Temperature')
ax1.tick_params('y')
ax1.legend(["TEMPERATURE"], loc=1)

ax2 = ax1.twinx()
ax2.plot(np.unique(df_train.index.year)[1:], US_soybean_prod)
ax2.set_ylabel('Soybean Production')
ax2.tick_params('y')
ax2.legend(["PRODUCTION"], loc=0)
ax1.grid()
fig, ax1 = plt.subplots(figsize=(15,6))
ax1.set_title("TEMPERATURE AND SOYBEAN PRODUCTION")
ax1.plot(np.unique(df_train.index.year)[1:], yearly_temp_avg, '-r')
ax1.set_xlabel('Year')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('Temperature')
ax1.tick_params('y')

ax2 = ax1.twinx()
ax2.plot(np.unique(df_train.index.year)[1:], US_soybean_prod)
ax2.set_ylabel('Soybean Production')
ax2.tick_params('y')
ax1.grid()


plt.figure(figsize=(20,18))
average_production = []
for i in range(0,22,2):
    plt.scatter(np.log10(df_train[production_origin[i]]), np.log10(df_train[production_origin[i+1]]))
    average_production.append(df_train[production_origin[i+1]])
plt.grid()
plt.title("Number of Area and Production " + str(int(np.average(average_production))))
plt.xlabel("Area of Soybean Plantation - Log Scale")
plt.ylabel("Production of Soybean Plantation - Log Scale")
plt.legend(["US", "Brazil", "Argentina", "China", 
            "India", "Paraguay", "Canada", 
            "Russia", "Central America", "Bolivia", "Africa"])
for i in range(0, 22, 2):
    fig, ax1 = plt.subplots(figsize=(15,6))
    ax1.set_title(production_origin[i].split("_")[0] + " AREA AND SOYBEAN PRODUCTION")
    ax1.plot(df_train[production_origin[i]], '-r')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Area')
    ax1.tick_params('y')
    ax1.legend(["AREA"], loc=0)

    ax2 = ax1.twinx()
    ax2.plot(df_train[production_origin[i+1]])
    ax2.set_ylabel('Soybean Production')
    ax2.tick_params('y')
    ax2.legend(["PRODUCTION"], loc=4)
    ax1.grid()

plt.figure(figsize=(15,8))
for i in range(0,22,2):
    plt.plot(df_train[production_origin[i+1]]/df_train[production_origin[i]])
plt.ylabel("Efficiency")
plt.xlabel("Year")
plt.title("LAND EFFICENCY")
plt.legend(["US", "Brazil", "Argentina", "China", 
            "India", "Paraguay", "Canada", 
            "Russia", "Central America", "Bolivia", "Africa"])
plt.grid()
fig, ax1 = plt.subplots(figsize=(15,6))
ax1.set_title("AVERAGE AREA AND SOYBEAN PRODUCTION")
ax1.plot(df_train[[production_origin[i] for i in range(0,22,2)]].mean(axis=1), '-r')
ax1.set_xlabel('Year')
ax1.set_ylabel('Area')
ax1.tick_params('y')
ax1.legend(["AREA"], loc=0)

ax2 = ax1.twinx()
ax2.plot(df_train[[production_origin[i+1] for i in range(0,22,2)]].mean(axis=1))
ax2.set_ylabel('Soybean Production')
ax2.tick_params('y')
ax2.legend(["PRODUCTION"], loc=4)
ax1.grid()

fig, ax1 = plt.subplots(figsize=(15,6))
ax1.set_title("SUM AREA AND SOYBEAN PRODUCTION")
ax1.plot(df_train[[production_origin[i] for i in range(0,22,2)]].sum(axis=1), '-r')
ax1.set_xlabel('Year')
ax1.set_ylabel('Area')
ax1.tick_params('y')
ax1.legend(["AREA"], loc=0)

ax2 = ax1.twinx()
ax2.plot(df_train[[production_origin[i+1] for i in range(0,22,2)]].sum(axis=1))
ax2.set_ylabel('Soybean Production')
ax2.tick_params('y')
ax2.legend(["PRODUCTION"], loc=4)
ax1.grid()
years = np.unique(df_train.index.year)
total_soybean = df_train[[production_origin[i+1] for i in range(0,22,2)]].sum(axis=1)

df_production = pd.DataFrame()
for i in range(0, 22, 2):
    df_production[production_origin[i+1]] = df_train[production_origin[i+1]] / total_soybean

proportion = pd.DataFrame([df_production.loc[str(i)].mean(axis=0) for i in years], index=years)
plt.figure(figsize=(20,40))
plt.title("PRPORTION OF SOYBEAN PRODUCTION")
sns.heatmap(proportion, fmt="f", annot=True, robust=True, cbar=False, linewidths=0.2, cmap="YlOrBr_r")
plt.figure(figsize=(14,6))
plt.title("Commodities Price Overtime")
plt.plot(df_train['Gold'])
plt.plot(df_train['USD'])
plt.plot(df_train['Oil'])
plt.plot(df_train['bean_settle'])
plt.xlabel("Year")
plt.ylabel("Price")
plt.grid()
plt.legend(["Gold", "USD", "Oil", "SOYBEAN"])

def plot_long(Series, title):
    plt.figure(figsize=(14,3))
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Price")
    plt.grid()
    plt.plot(Series)
    
plot_long(df_train['Gold'].pct_change(), "Gold Price Change in Percent Overtime")
plot_long(df_train['USD'].pct_change(), "USD Price Change in Percent Overtime")
plot_long(df_train['Oil'].pct_change(), "Oil Price Change in Percent Overtime")
plot_long(df_train['bean_settle'].pct_change(), "Soybean Price Change in Percent Overtime")
plt.figure(figsize=(14,3))
plt.title("Price Change in Overtime")
plt.xlabel("Year")
plt.ylabel("Change of Price in Percent")
plt.grid()
plt.plot(df_train['bean_settle'].loc['1986'].pct_change())
plt.plot(df_train['Gold'].loc['1986'].pct_change())
plt.plot(df_train['USD'].loc['1986'].pct_change())
plt.legend(["Soybean", "Gold", "USD"])

# We find the change percentage, fill the NaN with 0, and sort the index
gold_price = df_train["Gold"].loc['1986'].pct_change().fillna(0).sort_index()
usd_price = df_train["USD"].loc['1986'].pct_change().fillna(0).sort_index()
soy_price = df_train["bean_settle"].loc['1986'].pct_change().fillna(0).sort_index()

final_array = []
for i in [gold_price, usd_price, soy_price]:
    init = 10000
    arr = []
    for k in i:
        init += init * k
        arr.append(init)
    final_array.append(arr)

plt.figure(figsize=(14,3))
plt.title(str(10000) + " Decline/Growth in Overtime")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid()
plt.plot(gold_price.index, final_array[0])
plt.plot(gold_price.index, final_array[1])
plt.plot(gold_price.index, final_array[2])
plt.legend(["Gold", "USD", "Soybean"])

gold_price = df_train["Gold"].loc[:'1986'].pct_change().fillna(0).sort_index()
usd_price = df_train["USD"].loc[:'1986'].pct_change().fillna(0).sort_index()
soy_price = df_train["bean_settle"].loc[:'1986'].pct_change().fillna(0).sort_index()
oil_price = df_train["Oil"].loc[:'1986'].pct_change().fillna(0).sort_index()

final_array = []
for i in [gold_price, usd_price, soy_price, oil_price]:
    init = 10000
    arr = []
    for k in i:
        init += init * k
        arr.append(init)
    final_array.append(arr)

plt.figure(figsize=(14,3))
plt.title(str(10000) + " Decline/Growth in Overtime")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid()
plt.plot(gold_price.index, final_array[0])
plt.plot(gold_price.index, final_array[1])
plt.plot(gold_price.index, final_array[2])
plt.plot(gold_price.index, final_array[3])
plt.legend(["Gold", "USD", "Soybean", "Oil"])
years = np.unique(df_train.index.year)
average_soybean_price = [df_train["bean_settle"].loc[str(i)].mean() for i in years]
total_soybean_production = [df_train[[production_origin[i+1] for i in range(0,22,2)]].sum(axis=1).loc[str(i)].mean() for i in years]

fig, ax1 = plt.subplots(figsize=(15,6))
ax1.set_title("PRICE AND SOYBEAN PRODUCTION")
ax1.plot(years, average_soybean_price, '-r')
ax1.set_xlabel('Year')
ax1.set_ylabel('Price')
ax1.tick_params('y')
ax1.legend(["PRICE"], loc=2)

ax2 = ax1.twinx()
ax2.plot(years, total_soybean_production)
ax2.set_ylabel('Soybean Production')
ax2.tick_params('y')
ax2.legend(["PRODUCTION"], loc=4)
ax1.grid()
ax2.grid(linestyle=":")

plt.figure(figsize=(15,6))
plt.title("Soybean Demand")
plt.xlabel("Year")
plt.ylabel("Demand")
plt.grid()
plt.plot(years, np.multiply(total_soybean_production, average_soybean_price))
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go

init_notebook_mode(connected=True) #do not miss this line

trace0 = go.Scatter(x = df_train.bean_settle.index,
                   y = df_train.bean_settle,
                   mode = 'lines',
                   name = 'Soybean')

trace1 = go.Scatter(x = df_train.bean_settle.index,
                   y = df_train.meal_settle,
                   mode = 'lines',
                   name = 'Soymeal')

trace2 = go.Scatter(x = df_train.bean_settle.index,
                   y = df_train.soyoil_settle,
                   mode = 'lines',
                   name = 'Soyoil')

data = [trace0, trace1, trace2]
fig = go.Figure(data=data)

py.offline.iplot(fig)
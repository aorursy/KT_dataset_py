# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from fbprophet import Prophet
import operator

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/avocado.csv", parse_dates=['Date'])
train = df.copy() #Make a copy for a safe case
train.isnull().sum()
print(train.type.unique())
print(train.year.unique())
print(train.region.unique())
len(train.region.unique())
train['revenue'] = train['Total Volume'] * train['AveragePrice']
train.loc[train.type == "conventional", 'profit'] = (train["revenue"] * 15 ) / 100
train.loc[train.type == "organic", 'profit'] = (train["revenue"] * 45 ) / 100
def prophet(df_formatted, periods, draw=False):
    prop = Prophet()
    prop.fit(df_formatted)
    future_prop = prop.make_future_dataframe(periods=periods)
    forecast_prop = prop.predict(future_prop)
    if (draw == True):
        fig1_prop = prop.plot(forecast_prop)
        fig2_prop = prop.plot_components(forecast_prop)
    return forecast_prop 
df_TotalUS = train[train.region == 'TotalUS']
for type in ("organic", "conventional"):
    df_type = df_TotalUS[df_TotalUS.type == type]
    df_profit  = df_type[['Date', 'profit']]
    formatted_profit = df_profit.rename(columns={'Date':'ds', 'profit':'y'})
    forecast_profit = prophet(formatted_profit, 78, draw=True)
    
plt.show()
profit_by_region_and_type = {}

def get_profit_by_region_and_type(df, region):
    df_region = df[df.region == region]
    for type in ("organic", "conventional"):
        df_type = df_region[df_region.type == type]
        df_profit  = df_type[['Date', 'profit']]
        formatted_profit = df_profit.rename(columns={'Date':'ds', 'profit':'y'})
        forecast_profit = prophet(formatted_profit, 78)
        yhat_sum = forecast_profit.tail(78).yhat.sum()
        region_type_str = region + "_" + type
        profit_by_region_and_type[region_type_str] = yhat_sum

for region in train.region.unique():
    get_profit_by_region_and_type(train, region)

value_key = ((value, key) for (key,value) in profit_by_region_and_type.items())
sorted_value_key = sorted(value_key, reverse=True)
df_profit_net = pd.DataFrame(sorted_value_key, columns=["Total Profit", "RegionAndType"])
df_top_five = df_profit_net[2:7]
df_top_five
df_totalUS_Cal = df_profit_net.loc[0:2]
df_totalUS_Cal = df_totalUS_Cal.drop(axis=0, index=1)

for regionType in df_totalUS_Cal["RegionAndType"]:
    region = regionType.split("_")[0]
    type = regionType.split("_")[1]
    df_region_type = train[(train.type == type) & (train.region == region)] 
    df_corr = df_region_type.corr()
    df_corr_total_volume = df_corr['Total Volume'][0]
    print(regionType, df_corr_total_volume)

df_California = train[(train.region == "California")]

for dataset in (df_TotalUS, df_California):
    df_type = dataset[dataset.type == "conventional"]
    df_vol  = df_type[['Date', 'Total Volume']]
    formatted_vol = df_vol.rename(columns={'Date':'ds', 'Total Volume':'y'})
    forecast_vol = prophet(formatted_vol, 78, draw=True)
sns.boxplot(y="type", x="profit", data=df_California)
import sys

import os

from pprint import pprint

from tqdm import tqdm

import threading



import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

sns.set(color_codes=True)
folder="D:\\Data\\london_smartmeter\\dataset_kaggle\\"

folder="../input/"

folder_data_block="../input/dataset_kaggle/"
file='weather_daily_darksky.csv'#don't understand the loqdinf fqiling for the csv

df_weather_daily=pd.read_csv(folder+file)

df_weather_daily["time"]=pd.to_datetime(df_weather_daily["time"])

df_weather_daily=df_weather_daily.sort_values(["time"])

df_weather_daily["day"]=df_weather_daily.apply(lambda row:row["time"].strftime("%Y-%m-%d"),axis=1)

df_weather_daily["temperatureMean"]=df_weather_daily.apply(lambda row:(row["temperatureMax"]+row["temperatureMin"])/2,axis=1)

df_weather_daily=df_weather_daily.drop_duplicates(["day"])

df_weather_daily=df_weather_daily.set_index(("day"))

df_weather_daily.head()
# Works on the first file

file='block_0.csv'

df_block=pd.read_csv(folder_data_block+file)

df_block["tstp"]=pd.to_datetime(df_block["tstp"])

df_block["energy(kWh/hh)"]=pd.to_numeric(df_block["energy(kWh/hh)"],errors="coerce")

df_block=df_block.dropna()

df_block.head()
get_householdid=list(df_block["LCLid"].unique())
# Function to collect and analyse the data from one household

aggregation={

    "energy(kWh/hh)":{

        "min_energy":"min",

        "mean_energy":"mean",

        "max_energy":"max",

        "sum_energy":"sum",

        "count":"count"

    },

    "weekday":"first",

    "month":"first",

    "type_month":"first"  

}



def get_df_household(df_household):

    df_household["weekday"]=df_household.apply(lambda row:row["tstp"].weekday(),axis=1)

    df_household["day"]=df_household.apply(lambda row:row["tstp"].strftime("%Y-%m-%d"),axis=1)

    df_household["month"]=df_household.apply(lambda row:row["tstp"].strftime("%Y-%m"),axis=1)

    df_household["type_month"]=df_household.apply(lambda row:int(row["tstp"].strftime("%m")),axis=1)

    return df_household



def analyse_df_household(df_household):

    df_household=get_df_household(df_household)

    df_count=df_household.groupby(["day"]).agg(aggregation)

    df_cross=pd.concat([df_count,df_weather_daily],axis=1, join_axes=[df_count.index])

    return df_cross
#Works on a subset of the first block

dict_result={}

for household in tqdm(get_householdid[:10]):

    df_household=df_block[df_block["LCLid"]==household]

    df_cross=analyse_df_household(df_household)

    df_cross=df_cross.reset_index()

    df_cross["LCLid"]=[household]*len(df_cross)

    dict_result[household]=df_cross
# Make a plot to illustrathe the different behaviour for the first 10 housrholds in the first block file

fig,ax=plt.subplots(figsize=(12,12))

palette=sns.color_palette("Set2", len(list(dict_result.keys())))

for i,household in enumerate(dict_result):

    ax=plt.subplot(4,3,i+1)

    dict_result[household].plot(ax=ax,x="temperatureMean",y=('energy(kWh/hh)', 'sum_energy'),kind="scatter",color=palette[i],label=household)

plt.legend()

plt.show()
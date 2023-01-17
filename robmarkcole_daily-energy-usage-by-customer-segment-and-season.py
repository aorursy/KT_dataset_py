import pandas as pd
import numpy as np
import seaborn as sns
import os
import sys
import glob

data_dir = "/kaggle/input/smart-meters-in-london/daily_dataset/"
print(len(os.listdir(data_dir+'daily_dataset')))
os.listdir(data_dir+'daily_dataset')[:5]
daily_df = pd.read_csv(data_dir+'daily_dataset/block_71.csv')
daily_df['day'] = pd.to_datetime(daily_df['day'])
daily_df = daily_df.set_index('day')
daily_df.head()
len(daily_df["LCLid"].unique())
daily_df[daily_df["LCLid"]=="MAC000027"]["energy_sum"].plot(figsize=(20,6));
daily_df.reset_index().groupby("day").nunique()["LCLid"].plot(figsize=(20,6));

num_households_df = daily_df.reset_index().groupby("day").nunique()["LCLid"] # get the number of households on each day
energy_df = daily_df.reset_index().groupby("day").sum()["energy_sum"] # get the total energy usage per day

# normalise the energy usage to the number of households and plot
energy_per_household_df = pd.concat([num_households_df, energy_df], axis=1)
energy_per_household_df["normalised"] = energy_per_household_df["energy_sum"] / energy_per_household_df["LCLid"]
energy_per_household_df["normalised"].plot(figsize=(20,6));
info_df = pd.read_csv('/kaggle/input/smart-meters-in-london/informations_households.csv')
info_df.head()
# Helper to load a single file
def daily_to_df(file_path : str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df['day'] = pd.to_datetime(df['day'])
    df["year"] = df["day"].apply(lambda x : x.year)
    df["month"] = df["day"].apply(lambda x : x.month)
    df["dayofweek"] = df["day"].apply(lambda x : x.dayofweek)
    df["day_name"] = df["day"].apply(lambda x : x.day_name())
    df = df.merge(info_df, on="LCLid")
    df = df[df["year"].isin([2012, 2013])]
    return df[["LCLid", "day", "year", "month", "day_name", "Acorn_grouped", "energy_sum"]]

df = daily_to_df(data_dir+'daily_dataset/block_71.csv')
df.head()
all_daily_df = pd.DataFrame()

for i, file_path in enumerate(glob.glob(data_dir+'daily_dataset/*.csv')):
    all_daily_df = all_daily_df.append(daily_to_df(file_path))
    print(all_daily_df.shape)
all_daily_df = all_daily_df.drop_duplicates()
all_daily_df = all_daily_df.dropna()
all_daily_df.head()
y2013_df = all_daily_df[all_daily_df['year']==2013]
y2013_df.groupby("Acorn_grouped").count()["LCLid"]
y2013_df = y2013_df[y2013_df["Acorn_grouped"].isin(["Adversity", "Affluent", "Comfortable"])]
sum_y2013_df = pd.concat([y2013_df.groupby("Acorn_grouped").sum()["energy_sum"], y2013_df.groupby("Acorn_grouped").count()["LCLid"]], axis=1)
sum_y2013_df["normalised"] = sum_y2013_df["energy_sum"] / sum_y2013_df["LCLid"]
sum_y2013_df
sum_y2013_df["normalised"].plot.bar();
y2013_df.groupby("day_name").sum()["energy_sum"].sort_values().plot.bar();
y2013_df.groupby("month").sum()["energy_sum"].sort_values().plot.bar()

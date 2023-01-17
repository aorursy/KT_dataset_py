import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df_climbing = pd.read_csv("/kaggle/input/mount-rainier-weather-and-climbing-data/climbing_statistics.csv")

df_weather = pd.read_csv("/kaggle/input/mount-rainier-weather-and-climbing-data/Rainier_Weather.csv")
df_climbing.info()
df_climbing.describe()
df_climbing.head(10)
df_climbing[df_climbing["Date"] == "10/3/2015"]
# 2 for all succeeded

# 1 for partial success

# 0 no one succeed



df_climbing["all_group_climbed"] = df_climbing["Succeeded"].apply(lambda x: 2 if x > 0 else 0)

df_climbing.loc[ (df_climbing["Attempted"] > 0) & (df_climbing["Succeeded"] > 0) , "all_group_climbed"] = 1
df_climbing["all_group_climbed"].unique()
df_climbing[df_climbing["Succeeded"] > df_climbing["Attempted"]]
df_climbing["all_group_climbed"].value_counts().plot(kind='bar')
gb = df_climbing.groupby(["Route"]).sum().reset_index()



gb = gb[["Route","Attempted","Succeeded"]].melt("Route", var_name="a", value_name="b")

gb["b"] = gb["b"].apply(lambda x: 600 if x > 600 else x)

ax=sns.barplot(x='Route', y='b', hue='a', data=gb)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

print("")
df_weather.info()
df_weather.describe()
df_weather.head()
no_date_weather = df_weather[["Battery Voltage AVG","Temperature AVG","Relative Humidity AVG","Wind Speed Daily AVG","Wind Direction AVG","Solare Radiation AVG"]]

g = sns.pairplot(no_date_weather)
sns.heatmap(no_date_weather.corr())
df_climbing["Date and Route"] = df_climbing["Date"] + "#" + df_climbing["Route"]
# Doing some processing

new_df_climbing = df_climbing.groupby("Date and Route").mean().reset_index()

new_df_climbing = new_df_climbing.drop("all_group_climbed", axis=1)

new_df_climbing["Success Percentage"] = new_df_climbing["Succeeded"] / new_df_climbing["Attempted"]
# Split the Date and Route column into the individuals Date column and Route column

new = new_df_climbing["Date and Route"].str.split("#", n = 1, expand = True) 

new_df_climbing["Date"] = new[0]

new_df_climbing["Route"] = new[1]

new_df_climbing = new_df_climbing.drop("Date and Route", axis=1)
df = new_df_climbing.merge(df_weather, on="Date", how="inner")
df.head()
df.corr()["Success Percentage"]
sns.heatmap(df.corr())
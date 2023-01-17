import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os # operating system
import seaborn as sns # For plots
df = pd.read_csv("../input/BreadBasket_DMS.csv")
df = df[df["Item"] != "NONE"]
df["time"] = pd.to_datetime(df['Time'])
df["hour"] = df["time"].dt.hour
df["timeofday"] = "unknown"
df.loc[(df.hour <= 24), 'timeofday'] = "evening"
df.loc[(df.hour <= 15), 'timeofday'] = "day"
df.loc[(df.hour <= 11), 'timeofday'] = "morning"
df.drop("hour", axis=1, inplace = True)
print(df.head())
grouped = df.groupby("timeofday")
print(df.columns.values)
data_for_plots = []
for name,group in grouped:
    #print(group.head())
    #print(group.shape)
    value_counts = group["Item"].value_counts().to_frame().reset_index()
    allSum = value_counts["Item"].sum()
    value_counts["Item"] = value_counts["Item"] / allSum
    allSum = value_counts["Item"].sum()
    print(value_counts.columns.values)
    print(value_counts.head())
    value_counts = value_counts[:30]
    value_counts = value_counts.rename(index=str, columns={"index": "Object", "Item": "Relative probability [percent]"})
    print (name)
    data_for_plots.append(value_counts)
    
sns.barplot(y= "Object", x = "Relative probability [percent]", data = data_for_plots[2]).set_title("Morning")
sns.barplot(y= "Object", x = "Relative probability [percent]", data = data_for_plots[0]).set_title("Daytime")
sns.barplot(y= "Object", x = "Relative probability [percent]", data = data_for_plots[1]).set_title("Evening")
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import calendar

%matplotlib inline
#read csv file

weather = pd.read_csv("../input/GlobalTemperatures.csv")



#Filter out data older the 1850-01-01

weather = weather[weather["dt"] >= "1850-01-01"][["dt", "LandAverageTemperature"]]



#convert dt-column to a date-object

weather["dt"] = pd.to_datetime(weather["dt"])



#adds some new columns based on out date column

weather["year"] = weather["dt"].apply(lambda dt: dt.year)

weather["month"] = weather["dt"].apply(lambda dt: calendar.month_abbr[dt.month])



#calculate the mean for a given month 

total_mean_temps = weather[["month", "LandAverageTemperature"]].groupby("month").mean()



weather["total average temperature"] = weather["month"].apply(lambda m: total_mean_temps.loc[m])



#New column for determine wether a month has increased or decreased in temp. when compared to the mean for that month

weather["diff"] = weather["LandAverageTemperature"] - weather["total average temperature"]



result = weather[["year", "month", "diff"]].groupby(by=["year", "month"]).sum().unstack().xs("diff", axis=1)[::-1]

result = result.reindex_axis(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], axis=1)





#plot everything

plt.figure(figsize=(12, 8))

sns.heatmap(result, yticklabels=8, cmap="gnuplot2", vmin=-3)

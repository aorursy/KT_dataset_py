%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import scipy.stats as stats

import os
crashes = pd.read_csv("../input/airplane-crashes-since-1908/Airplane_Crashes_and_Fatalities_Since_1908.csv")
airlines = pd.read_csv("../input/airline-safety/airline-safety.csv")
crashes.head()
crashes.info()
100 * crashes.isnull().sum()/len(crashes)
del crashes["Time"]

del crashes["Flight #"]

del crashes["Route"]

del crashes["cn/In"]

del crashes["Registration"]

del crashes["Ground"]
crashes.dropna(subset=["Location","Operator","Type","Aboard","Fatalities"], inplace=True)
crashes["Date"] = pd.to_datetime(crashes["Date"])
crashes.head()
crashes.info()
100 * crashes.isnull().sum()/len(crashes)
airlines.head()
airlines.info()
100 * airlines.isnull().sum()/len(airlines)
crashes["Survival Rate"] = 100 * (crashes["Aboard"] - crashes["Fatalities"]) / crashes["Aboard"]

crashes.head()
data_nobs = len(crashes["Survival Rate"])

data_mean = crashes["Survival Rate"].mean()

data_min = crashes["Survival Rate"].min()

data_max = crashes["Survival Rate"].max()

data_var = crashes["Survival Rate"].var()

data_skew = crashes["Survival Rate"].skew()

data_kurtosis = crashes["Survival Rate"].kurtosis()



print("Survival Rate Stats:")

print("Nobs: {}".format(round(data_nobs,2)))

print("Mean: {}".format(round(data_mean,2)))

print("Min: {}".format(round(data_min,2)))

print("Max: {}".format(round(data_max,2)))

print("Variance: {}".format(round(data_var,2)))

print("Skewness: {}".format(round(data_skew,2)))

print("Kurtosis: {}".format(round(data_kurtosis,2)))

yearly_survival = crashes[["Date","Survival Rate"]].groupby(crashes["Date"].dt.year).agg(["mean"])

yearly_survival.plot(legend=None)

plt.ylabel("Average Survival Rate, %")

plt.xlabel("Year")

plt.title("Average Survival Rate per Year")

plt.xticks([x for x in range(1908,2009,10)], rotation='vertical')

plt.axhline(y=data_mean, color='r', linestyle='-')

plt.show()
airlines.incidents_85_99.hist(label="1985 - 1999", alpha = 0.5)

airlines.incidents_00_14.hist(label="2000 - 2015", alpha = 0.5)

plt.legend(loc="upper right")

plt.xlabel("Accidents per Airline")

plt.ylabel("Frequency")

plt.title("Histogram of Accidents per Airline")

plt.show()
#1985 - 1999: Accidents

data = airlines.incidents_85_99

data_nobs = len(data)

data_mean = data.mean()

data_min = data.min()

data_max = data.max()

data_var = data.var()

data_skew = data.skew()

data_kurtosis = data.kurtosis()



print("1985 - 1999 Accidents Stats:")

print("Nobs: {}".format(round(data_nobs,2)))

print("Mean: {}".format(round(data_mean,2)))

print("Min: {}".format(round(data_min,2)))

print("Max: {}".format(round(data_max,2)))

print("Variance: {}".format(round(data_var,2)))

print("Skewness: {}".format(round(data_skew,2)))

print("Kurtosis: {}".format(round(data_kurtosis,2)))

#2000 - 2014: Accidents

data = airlines.incidents_00_14

data_nobs = len(data)

data_mean = data.mean()

data_min = data.min()

data_max = data.max()

data_var = data.var()

data_skew = data.skew()

data_kurtosis = data.kurtosis()



print("2000 - 2014 Accidents Stats:")

print("Nobs: {}".format(round(data_nobs,2)))

print("Mean: {}".format(round(data_mean,2)))

print("Min: {}".format(round(data_min,2)))

print("Max: {}".format(round(data_max,2)))

print("Variance: {}".format(round(data_var,2)))

print("Skewness: {}".format(round(data_skew,2)))

print("Kurtosis: {}".format(round(data_kurtosis,2)))
t_stat, p_val = stats.wilcoxon(airlines.incidents_85_99, airlines.incidents_00_14)

print("t-stat: {}. p-val: {}.".format(round(t_stat,3),round(p_val,3)))
airlines.fatalities_85_99.hist(label="1985 - 1999", alpha = 0.5)

airlines.fatalities_00_14.hist(label="2000 - 2015", alpha = 0.5)

plt.legend(loc="upper right")

plt.xlabel("Fatalities per Airline")

plt.ylabel("Frequency")

plt.title("Histogram of Fatalities per Airline")

plt.show()
#1985 - 1999: Fatalities

data = airlines.fatalities_85_99

data_nobs = len(data)

data_mean = data.mean()

data_min = data.min()

data_max = data.max()

data_var = data.var()

data_skew = data.skew()

data_kurtosis = data.kurtosis()



print("1985 - 1999 Fatalities Stats:")

print("Nobs: {}".format(round(data_nobs,2)))

print("Mean: {}".format(round(data_mean,2)))

print("Min: {}".format(round(data_min,2)))

print("Max: {}".format(round(data_max,2)))

print("Variance: {}".format(round(data_var,2)))

print("Skewness: {}".format(round(data_skew,2)))

print("Kurtosis: {}".format(round(data_kurtosis,2)))

#2000 - 2014: Fatalities

data = airlines.fatalities_00_14

data_nobs = len(data)

data_mean = data.mean()

data_min = data.min()

data_max = data.max()

data_var = data.var()

data_skew = data.skew()

data_kurtosis = data.kurtosis()



print("2000 - 2014 Fatalities Stats:")

print("Nobs: {}".format(round(data_nobs,2)))

print("Mean: {}".format(round(data_mean,2)))

print("Min: {}".format(round(data_min,2)))

print("Max: {}".format(round(data_max,2)))

print("Variance: {}".format(round(data_var,2)))

print("Skewness: {}".format(round(data_skew,2)))

print("Kurtosis: {}".format(round(data_kurtosis,2)))
t_stat, p_val = stats.wilcoxon(airlines.fatalities_85_99, airlines.fatalities_00_14)

print("t-stat: {}. p-val: {}.".format(round(t_stat,3),round(p_val,3)))
airlines.fatal_accidents_85_99.hist(label="1985 - 1999", alpha = 0.5)

airlines.fatal_accidents_00_14.hist(label="2000 - 2015", alpha = 0.5)

plt.legend(loc="upper right")

plt.xlabel("Fatal Accidents per Airline")

plt.ylabel("Frequency")

plt.title("Histogram of Fatal Accidents per Airline")

plt.show()
#1985 - 1999: Fatal Accidents

data = airlines.fatal_accidents_85_99

data_nobs = len(data)

data_mean = data.mean()

data_min = data.min()

data_max = data.max()

data_var = data.var()

data_skew = data.skew()

data_kurtosis = data.kurtosis()



print("1985 - 1999 Fatal Accidents Stats:")

print("Nobs: {}".format(round(data_nobs,2)))

print("Mean: {}".format(round(data_mean,2)))

print("Min: {}".format(round(data_min,2)))

print("Max: {}".format(round(data_max,2)))

print("Variance: {}".format(round(data_var,2)))

print("Skewness: {}".format(round(data_skew,2)))

print("Kurtosis: {}".format(round(data_kurtosis,2)))

#2000 - 2014: Fatal Accidents

data = airlines.fatal_accidents_00_14

data_nobs = len(data)

data_mean = data.mean()

data_min = data.min()

data_max = data.max()

data_var = data.var()

data_skew = data.skew()

data_kurtosis = data.kurtosis()



print("2000 - 2014 Fatal Accidents Stats:")

print("Nobs: {}".format(round(data_nobs,2)))

print("Mean: {}".format(round(data_mean,2)))

print("Min: {}".format(round(data_min,2)))

print("Max: {}".format(round(data_max,2)))

print("Variance: {}".format(round(data_var,2)))

print("Skewness: {}".format(round(data_skew,2)))

print("Kurtosis: {}".format(round(data_kurtosis,2)))

t_stat, p_val = stats.wilcoxon(airlines.fatal_accidents_85_99, airlines.fatal_accidents_00_14)

print("t-stat: {}. p-val: {}.".format(round(t_stat,3),round(p_val,3)))
airlines["total_incidents"] = airlines["incidents_85_99"] + airlines["incidents_00_14"]

airlines["total_fatal_accidents"] = airlines["fatal_accidents_85_99"] + airlines["fatal_accidents_00_14"]

airlines["total_fatalities"] = airlines["fatalities_85_99"] + airlines["fatalities_00_14"]
del airlines["incidents_85_99"]

del airlines["incidents_00_14"]

del airlines["fatal_accidents_85_99"]

del airlines["fatal_accidents_00_14"]

del airlines["fatalities_85_99"]

del airlines["fatalities_00_14"]
airlines["norm_total_incidents"] = airlines["total_incidents"] / airlines["avail_seat_km_per_week"] * 1000000

airlines["norm_total_fatal_accidents"] = airlines["total_fatal_accidents"] / airlines["avail_seat_km_per_week"] * 1000000

airlines["norm_total_fatalities"] = airlines["total_fatalities"] / airlines["avail_seat_km_per_week"] * 1000000
del airlines["total_incidents"]

del airlines["total_fatal_accidents"]

del airlines["total_fatalities"]
#Normalized Accidents

data = airlines[["airline","norm_total_incidents"]]

data = data.sort_values(by="norm_total_incidents", ascending=False).head()

data.set_index("airline", inplace=True)

data.norm_total_incidents.plot(kind="bar")

plt.title("Airlines with Most Normalized Accidents")

plt.xlabel("Airline")

plt.ylabel("Normalized Accidents")

plt.show()
#Normalized Fatal Accidents

data = airlines[["airline","norm_total_fatal_accidents"]]

data = data.sort_values(by="norm_total_fatal_accidents", ascending=False).head()

data.set_index("airline", inplace=True)

data.norm_total_fatal_accidents.plot(kind="bar")

plt.title("Airlines with Most Normalized Fatal Accidents")

plt.xlabel("Airline")

plt.ylabel("Normalized Fatal Accidents")

plt.show()
#Normalized Fatalities

data = airlines[["airline","norm_total_fatalities"]]

data = data.sort_values(by="norm_total_fatalities", ascending=False).head()

data.set_index("airline", inplace=True)

data.norm_total_fatalities.plot(kind="bar")

plt.title("Airlines with Most Normalized Fatalities")

plt.xlabel("Airline")

plt.ylabel("Normalized Fatalities")

plt.show()
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns


avocados = pd.read_csv("../input/avocado.csv")
avocados.rename(columns = {"Unnamed: 0" : "Avocado ID", "AveragePrice" : "Average Price"}, inplace=True) #because consistent naming makes things a lot easier.
#LEARNING: also check for inconsistent casing of column names
print(avocados.columns)
date_conversion = pd.to_datetime(avocados["Date"], format='%Y-%m-%d')
avocados["old_date"] = avocados["Date"]
avocados["Date"] = date_conversion

avocados.describe(include="all").round(2)
grp = avocados['Average Price'].groupby([avocados["Date"], avocados["region"]]).mean()
index_values = pd.Series(grp.index.values).apply(pd.Series)
group_by_date_and_region = pd.DataFrame(data={"date" : index_values[0], "region" : index_values[1], "average_price" : grp.values})

dates = avocados.Date.unique()
dates.sort()
region = avocados.region.unique()

region.sort() #alpha sort the regions
leaderboard = np.zeros(len(region)) #put in a row of zeros to enable the vstacking in the loop below

for date in dates:
    
    date_slice = group_by_date_and_region.loc[group_by_date_and_region["date"] == date]
    leaderboard_for_date = pd.DataFrame(data={"region" : date_slice["region"], "rank" : date_slice["average_price"].rank(), 
                                        "average_price" : date_slice["average_price"]})
    leaderboard_for_date = leaderboard_for_date.sort_values("region")
    leaderboard_for_date = leaderboard_for_date["rank"].values.reshape(-1, 1).transpose()
    leaderboard = np.vstack((leaderboard, leaderboard_for_date))

leaderboard = np.delete(leaderboard, 0, 0) #remove the row of zeros initially inserted to allow vstacking
leaderboard = pd.DataFrame(columns=region, data=leaderboard)
leaderboard["date"] = dates
leaderboard = leaderboard.set_index("date")
dates = leaderboard.index.values
regions = avocados.region.unique()

fig, ax = plt.subplots(1, 1, figsize=(25, 12))

for region in regions:
    plt.plot(dates, leaderboard[region].values)
    
plt.title("Price leadership of regions over time", fontsize=20)
plt.show()
#split the date values into year and date. Group on date

avocados["time_tuple"] = avocados["Date"]
avocados["time_tuple"] = avocados["time_tuple"].apply(lambda x: x.timetuple())
avocados["year, month"] = avocados["time_tuple"].apply(lambda x: pd.Period((str(x[0]) + "-" + str(x[1]))))


grp = avocados['Average Price'].groupby([avocados["year, month"], avocados["region"]]).mean()
index_values = pd.Series(grp.index.values).apply(pd.Series)
group_by_date_and_region = pd.DataFrame(data={"year, month" : index_values[0], "region" : index_values[1], "average_price" : grp.values})

dates = avocados["year, month"].unique()
dates.sort()
region = avocados.region.unique()

region.sort() #alpha sort the regions
leaderboard = np.zeros(len(region)) #put in a row of zeros to enable the vstacking in the loop below

for date in dates:
    
    date_slice = group_by_date_and_region.loc[group_by_date_and_region["year, month"] == date]
    leaderboard_for_date = pd.DataFrame(data={"region" : date_slice["region"], "rank" : date_slice["average_price"].rank(), 
                                        "average_price" : date_slice["average_price"]})
    leaderboard_for_date = leaderboard_for_date.sort_values("region")
    leaderboard_for_date = leaderboard_for_date["rank"].values.reshape(-1, 1).transpose()
    leaderboard = np.vstack((leaderboard, leaderboard_for_date))

leaderboard = np.delete(leaderboard, 0, 0) #remove the row of zeros initially inserted to allow vstacking
leaderboard = pd.DataFrame(columns=region, data=leaderboard)
leaderboard["year, month"] = dates
leaderboard = leaderboard.set_index("year, month")
dates = leaderboard.index.values
regions = avocados.region.unique()

fig, ax = plt.subplots(1,1, figsize=(25, 12))
x = np.arange(0,len(dates),1)
ax.set_xticks(x)
ax.set_xticklabels(dates)
for region in regions:
    plt.plot(x,leaderboard[region].values)
plt.xticks(x, rotation='vertical')
plt.title("Price leadership of regions averaged per month", fontsize=20)
plt.show()

grp = leaderboard.mean(axis=0).sort_values()
top = grp.head(5).index.values

dates = leaderboard.index.values
regions = top

fig, ax = plt.subplots(1,1, figsize=(30, 20))
x = np.arange(0,len(dates),1)
ax.set_xticks(x)
ax.set_xticklabels(dates)
artists = []
for region in regions:
    #plt.plot(dates, leaderboard[region].values)
    plt.plot(x,leaderboard[region].values, label=region)
    
plt.xlabel("month", fontsize=20)
plt.xticks(rotation='vertical', fontsize=20)
plt.ylabel("leadership rank", fontsize=20)
plt.yticks(np.arange(0, 45, 1.0))
plt.legend(fontsize=20)
plt.title("Top five price leaders averaged per month", fontsize=20)
plt.show()
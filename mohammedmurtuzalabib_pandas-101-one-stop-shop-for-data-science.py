import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt
list_of_dicts = [ 

     {"name": "Ginger", "breed": "Dachshund", "height_cm": 22,"weight_kg": 10, "date_of_birth": "2019-03-14"},

    {"name": "Scout", "breed": "Dalmatian", "height_cm": 59,"weight_kg": 25, "date_of_birth": "2019-05-09"}

]

new_dogs = pd.DataFrame(list_of_dicts)

new_dogs
dict_of_lists = { 

     "name": ["Ginger", "Scout"], 

     "breed": ["Dachshund", "Dalmatian"], 

     "height_cm": [22, 59], 

     "weight_kg": [10, 25], 

     "date_of_birth": ["2019-03-14","2019-05-09"]  } 

new_dogs = pd.DataFrame(dict_of_lists) 

new_dogs
# read CSV from using pandas

avocado = pd.read_csv("../input/avocado-prices/avocado.csv")

# print the first few rows of the dataframe

avocado.head()
# read CSV from using pandas and assigning Date as index of the dataframe

avocado = pd.read_csv("../input/avocado-prices/avocado.csv",parse_dates=True, index_col='Date')

# print the first few rows of the dataframe

avocado.head()
avocado = avocado.reset_index(drop=True)

avocado.head()
avocado.to_csv("test_write.csv")
avocado = pd.read_csv("../input/avocado-prices/avocado.csv")

avocado.head()
avocado.tail(10)
avocado.info()
print(avocado.shape)
avocado.describe()
avocado.values
print(avocado.columns)
even = pd.Series([2,4,6,8,10])

odd = pd.Series([1,3,5,7,9])



res = even.append(odd)

res
res.reset_index(drop=True)
# sort values based on "AveragePrice" (ascending) and "year" (descending)

avocado.sort_values(["AveragePrice", "year"], ascending=[True, False]) 
# Subsetting columns

avocado["AveragePrice"]
# Subsetting multiple columns

avocado[["AveragePrice","Date"]]
# Subsetting rows

avocado["AveragePrice"]<1
# This will print only the rows with price < 1

avocado[avocado["AveragePrice"]<1]
# it will print all the rows with "type" = "organic"

avocado[avocado["type"]=="organic"]
# it will print all the rows with "Date" <= 2015-02-04

avocado[avocado["Date"]<="2015-02-04"]
# it will print all the rows with "Date" before 2015-02-04 and "type" == "organic"

avocado[(avocado["Date"]<"2015-02-04") & (avocado["type"]=="organic")]
# subset the avocado in the region Boston or SanDiego

regionFilter = avocado["region"].isin(["Boston", "SanDiego"])

avocado[regionFilter]
# subset the avocado in the region Boston or SanDiego in the year 2016 or 2017

regionFilter = avocado["region"].isin(["Boston", "SanDiego"])

yearFilter = avocado["year"].isin(["2016", "2017"])

avocado[regionFilter & yearFilter]
avocado.isna()
avocado.isna().any()
avocado.isna().sum()
# Luckily we don't have any NaN but if we have we can use any of the two methods



avocado.dropna()



# ****  OR  ****



meanVal = avocado["AveragePrice"].mean()

avocado.fillna(meanVal)
avocado["AveragePricePer100"] = avocado["AveragePrice"] * 100

avocado
avocado.drop(["AveragePricePer100"],axis = 1)
# mean of the AveragePrice of avocado

avocado["AveragePrice"].mean()
avocado["Date"].max()
def pct30(column):     

    #return the 0.3 quartile

    return column.quantile(0.3)

def pct50(column):     

    #return the 0.5 quartile

    return column.quantile(0.5)



avocado[["AveragePrice","Total Bags"]].agg([pct30,pct50])
temp = avocado.drop_duplicates(subset=["year"])

temp
# count number of avocado in each year in descending order

avocado["year"].value_counts(sort=True, ascending = False)
# group by multiple columns and perform multiple summary statistic operations

avocado.groupby(["year","type"])["AveragePrice"].agg([min,max,np.mean,np.median])
# this is the same table we build in the previous cell but using pivot table

avocado.pivot_table(index=["year","type"], aggfunc=[min,max,np.mean,np.median], values="AveragePrice")
regionIndex = avocado.set_index(["region"])

regionIndex
# Insted of doing this

avocado[avocado["region"].isin(["Albany", "WestTexNewMexico"])]
# we can simply do

regionIndex.loc[["Albany", "WestTexNewMexico"]]
avocado["AveragePrice"].hist(bins=20)

plt.show()
regionFilter = avocado.groupby("region")["AveragePrice"].mean().head(10)

regionFilter
regionFilter.plot(kind = "bar",rot=45,title="Average price in 10 regions")
avocado.plot(x="AveragePrice", y="Total Volume", kind="scatter")
# subtract AveragePrice with AveragePrice :P

# Dah its 0

avocado["AveragePrice"].sub(avocado["AveragePrice"]) 
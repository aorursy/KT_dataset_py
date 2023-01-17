import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
result = reviews["points"].median()
print(result)
check_q1(result)
result = reviews["country"].unique()
print(result)
check_q2(result)
result = reviews["country"].value_counts()
print(result)
check_q3(result)
price_median = reviews["price"].median()
print(price_median)
data = reviews["price"].map(lambda price: price - price_median)
print(data)
check_q4(data)
price_median = reviews["price"].median()
print(price_median)
data = reviews["price"].apply(lambda price: price - price_median) # returns 1 column
print(data)
check_q5(data)

# ------------------

def remedians(srs):
    srs.price = srs.price - price_median
    return srs

data2 = reviews.apply(remedians, axis='columns') # returns the data frame
print(reviews["price"]) # 'reviews' is not changed
print(data2["price"])

check_q5(data2["price"])

points_to_price = reviews["points"] / reviews["price"]
max_ptp = points_to_price.idxmax() # index of the position with maximum value
wine = reviews.loc[max_ptp]["title"]

print(wine)
check_q6(wine)


# To understand the answer:
# "tropical" in reviews["description"][0] >> Returns boolean if "tropical" is present on descritption
# The 'map' function makes an array with booleans, indicating that the string contains the specified word
# The 'value_counts' counts how many times the value is True and False

tropical_wine = reviews.description.map(lambda r: "tropical" in r).value_counts()
fruity_wine = reviews.description.map(lambda r: "fruity" in r).value_counts()

# tropical_wine[True] is the number of times "tropical" appears on description
# the line below makes a Series containing the number of times "tropical" and "fruity" appeared on description
result  = pd.Series([tropical_wine[True], fruity_wine[True]], index=['tropical', 'fruity'])

check_q7(result)

# selecting notnull values
filtered_data = reviews.loc[(reviews["country"].notnull()) & (reviews["variety"].notnull())]

def concatenate(dataframe):
    dataframe = dataframe["country"] + " - " + dataframe["variety"]
    return dataframe

# concatenate the string 'country - variety'
# it could be a lambda: >> filtered_data.apply(lambda srs: srs.country + " - " + srs.variety, axis='columns')
data_map = filtered_data.loc[:, ["country","variety"]].apply(concatenate, axis="columns")

# create a Series counting the occurrences >> value_counts() returns a pd.Series()
series = data_map.value_counts()
print(series)

check_q8(series)

import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
res = reviews.points.median()
print(res)
check_q1(res)
# Your code here
res =reviews.country.unique()
print(res)
check_q2(res)
# Your code here
res = reviews.country.value_counts()
print(res)
check_q3(res)
# Your code here
price_median = reviews.price.median()
res = reviews.price.map(lambda x: x - price_median)
print(res)
check_q4(res)
# Your code here
def points_to_price_ratio(series):
    try:
        series.points = series.points / series.price
    except ZeroDivisionError as e:
        series.points = -999
    return series

res = reviews.copy()
res = res.apply(points_to_price_ratio, axis=1)

res = res.loc[res.points.notnull(),["points","title"]]

res = res.loc[res.points.argmax()].title        

print(res)        
print(check_q6(res))

# Your code here
res = reviews.copy()
tropical_count = res.description.map(lambda x: True if "tropical" in x else False)
tropical_count.name = "Tropical"
tr = tropical_count.value_counts()
print(tr)
print("-------------------------")
fruity_count = res.description.map(lambda x: True if "fruity" in x else False)
# Your code here
res = reviews.copy()
tropical_count = res.description.map(lambda x: True if "tropical" in x else False)
tropical_count.name = "Tropical"
tr = tropical_count.value_counts()
print(tr)
print("-------------------------")
fruity_count = res.description.map(lambda x: True if "fruity" in x else False)
fruity_count.name = "Fruity"
fr = fruity_count.value_counts()
print(fr)
print("-------------------------")
res = pd.Series([tr[1],fr[1]], index = ["tropical","fruity"])
print(res)
print(check_q7(res)) 

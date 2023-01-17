import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
print(reviews.shape)
print(reviews.head())
reviews.describe()
res = reviews.points.median()
print(res)
print(check_q1(res))
res = reviews.country.unique()
print(res)
print(check_q2(res))
res = reviews.country.value_counts()
print(res)
print(check_q3(res))
# res = reviews.copy()
# median_price = res.price.median()
# res = res.price.map(lambda k: k - median_price)
price_median = reviews.price.median()
res = reviews.price.map(lambda x: x - price_median)
print(res)
print(check_q4(res))
def points_to_price_ratio(series):
    try:
        series.points = series.points / series.price
    except ZeroDivisionError as e:
        series.points = -999
    return series
res = reviews.copy()
res = res.apply(points_to_price_ratio, axis=1)
# print(res)
res = res.loc[res.points.notnull(),["points","title"]]
# print(res)
res = res.loc[res.points.argmax()].title
print(res)
print(check_q6(res))
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
res = pd.Series([tr[1],fr[1]], index=["tropical","fruity"])
print(res)
print(check_q7(res)) 
res = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
# res["country_variety"]=0
def add_new_col(series):
    series.country = series.country + " - " + series.variety
    return series
# res = res.apply(lambda x: x.country+" - "+x.variety, axis=1)
res = res.apply(add_new_col, axis=1)
res = res.country.value_counts()
print(res)
print(check_q8(res))
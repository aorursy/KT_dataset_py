import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(20)
reviews.head()
q1_median = reviews.points.median()
print(q1_median)
check_q1(q1_median)
q2_countries = reviews.country.unique()
print(q2_countries)
check_q2(q2_countries)
q3 = reviews.country.value_counts()
print(q3[:5])
check_q3(q3)
q4_price_median = reviews.price.median()
print(q4_price_median)
print(reviews.price[0:4])
q4 = reviews.price.map(lambda p: p - q4_price_median)
print(q4[0:4])
check_q4(q4)
q5_ratios = (reviews.points / reviews.price)
print(q5_ratios[0:5])

q5_max_index = q5_ratios.idxmax()
print(q5_max_index)

q5_title = reviews.loc[q5_max_index, 'title']
print(q5_title)

check_q5(q5_title)
# is_tropical = reviews.loc["tropical" in reviews.description]
# is_fruity = reviews.loc["fruity" in reviews.description]
# print(is_tropical.description)

has_tropical = reviews.description.map(lambda d: "tropical" in d).value_counts()
print(has_tropical)
has_fruity = reviews.description.map(lambda d: "fruity" in d).value_counts()
print(has_fruity)
q6_series = pd.Series(data=[has_tropical[True], has_fruity[True]], index=['tropical','fruity'])
print(q6_series)
check_q6(q6_series)
# Get reviews with country and variety provided
q7_subset = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
# print(q7_subset[0:5])
country_variety_set = q7_subset.apply(lambda srs: srs.country + ' - ' + srs.variety, axis="columns")
print(country_variety_set[0:5])

q7_solution = country_variety_set.value_counts()
check_q7(q7_solution)
import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
median_price = reviews.points.median()
# Your code here
countries = reviews.country.unique()
check_q2(countries)
# Your code here
df3 = reviews.country.value_counts()
check_q3(df3)
# Your code here
median_price = reviews.price.median()
df4 = reviews.price.map(lambda v: v - median_price)
check_q4(df4)
# Your code here
best_bargain = reviews.points/reviews.price
expected = reviews.loc[best_bargain.argmax()].title
print(expected)
check_q5(expected)
# Your code here
des_tropical = reviews.description.map(lambda r:"tropical" in r).value_counts()
des_fruity = reviews.description.map(lambda r:"fruity" in r).value_counts()
df6 = pd.Series([des_tropical[True],des_fruity[True]],index = ['tropical','fruity'])
print(df6)
print(type(df6))
check_q6(df6)
# Your code here
df7 = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
#sl8 = df7.country.map(str) + df7.variety.map(str)
df7 = df7.apply(lambda srs: srs.country + " - " + srs.variety,axis = 'columns')
count = df7.value_counts()
check_q7(count)
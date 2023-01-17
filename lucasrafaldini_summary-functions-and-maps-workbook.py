import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
df = reviews.points.median()
print(df)
check_q1(df)
df = reviews.country
print(df.head())
check_q2(df)
df = reviews.country.value_counts()

print(df.head())
check_q3(df)
reviews_median = reviews.price.median()
df = reviews.price.map(lambda v: v - reviews_median)
print(reviews_median)
check_q4(df)
median_price = reviews.price.median()
df = reviews.price.apply(lambda v: v - median_price)

df_DataFrame = pd.DataFrame({'Wines': reviews.title, 'How much more or less expensive than the median price': df}).fillna(0)

print(df_DataFrame.head(), '\n')
print(df.head())
check_q5(df)
ans = reviews.loc[(reviews.points / reviews.price).idxmax].title
print(ans)

check_q6(ans)
tropical_wine = reviews.description.map(lambda r: "tropical" in r).value_counts()
fruity_wine = reviews.description.map(lambda r: "fruity" in r).value_counts()
ans = pd.Series([tropical_wine[True], fruity_wine[True]], index=['tropical', 'fruity'])


check_q7(ans)
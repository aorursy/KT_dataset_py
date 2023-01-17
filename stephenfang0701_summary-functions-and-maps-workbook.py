import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
reviews.points.median()
# Your code here
print(reviews.country.unique())
# Your code here
reviews.country.value_counts().idxmax()
# reviews.groupby('country').size().idxmax()
reviews.head()
# Your code here
median_p = reviews.price.median()
reviews.price.map(lambda p: p - median_p)
# Your code here
median_p = reviews.price.median()
reviews.apply(lambda row: row.price-median_p, axis=1)
# Your code here
reviews.loc[(reviews.points/reviews.price).idxmax()].title
answer_q6()
# Your code here
tropical_count = reviews.description.map(lambda r: "tropical" in r).value_counts()
fruity_count = reviews.description.map(lambda x: "fruity" in x).value_counts()
count = pd.Series([tropical_count[True], fruity_count[True]], index=['tropical', 'fruity'])
check_q6(count)
reviews.head()
print(reviews.country.isnull().value_counts())
print(reviews.variety.isnull().value_counts())
answer_q7()
# Data Cleansing: drop the entries, which contain null value of country or variety
df = reviews.xs(['country', 'variety'], axis=1).dropna()
# df = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]

ans = df.apply(lambda r: r['country'] + ' - ' + r['variety'], axis=1)
check_q7(ans.value_counts())
import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
df = reviews.points.median()
print(check_q1(df))
df = reviews.country.unique()
print(check_q2(df))
df = reviews.country.value_counts()
print(check_q3(df))
med = reviews.price.median()
df = reviews.price.map(lambda p: p - med)
print(check_q4(df))
# tough
def points_ratio(series):
    series.points = series.points / series.price
    return series

df = reviews.apply(points_ratio, axis=1)
res = df.loc[df.points.argmax()].title
print(check_q6(res))

# easy
highest_ratio = (reviews.points / reviews.price).idxmax()
df = reviews.iloc[highest_ratio].title
print(check_q6(df))

df = reviews.description.map(lambda x: True if 'tropical' in x else False)
tr = df.value_counts()

df = reviews.description.map(lambda x: True if 'fruity' in x else False)
fr = df.value_counts()

df = pd.Series([tr[True], fr[True]], index=['tropical', 'fruity'])
print(df)
print(check_q7(df))
df = reviews[(reviews.country.notnull()) & (reviews.variety.notnull())]
df = df.country + ' - ' + df.variety
df = pd.Series(df.value_counts(), name= 'country')
print(df)
print(check_q8(df))
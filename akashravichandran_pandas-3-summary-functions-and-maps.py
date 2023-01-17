import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
df = reviews['points'].median()
print(df)
check_q1(df)
# Your code here
df = reviews['country'].unique()
print(df)
print(check_q2(df))
# Your code here
df = reviews['country'].value_counts()
print(df)
print(check_q3(df))
# Your code here
median = reviews['price'].median()
copy = reviews.copy()
df = copy['price'].map(lambda v: v - median)
print(df)
print(check_q4(df))
# Your code here
def ppratio(series):
    try:
        series['points'] = series['points'] / series['price']
    except:
        series['points'] = -5
    return series

res = reviews.copy()
res['price'] = res['price'].fillna(0.0)
res = res.apply(ppratio, axis=1)
res = res.loc[res['points'].idxmax()].title
print(res)
print(check_q6(res))
# Your code here
res = reviews.copy()
tropical_wine = res['description'].map(lambda r: "tropical" in r).value_counts()
fruity_wine = res['description'].map(lambda r: "fruity" in r).value_counts()
df = pd.Series([tropical_wine[True], fruity_wine[True]], index=['tropical', 'fruity'])
print(df)
print(check_q7(df))
# Your code here
res = reviews.copy()
ans = res.loc[(reviews.country.notnull()) & (res.variety.notnull())]
ans = ans.apply(lambda srs: srs.country + " - " + srs.variety, axis='columns')
df = ans.value_counts()
print(df)
print(check_q8(df))
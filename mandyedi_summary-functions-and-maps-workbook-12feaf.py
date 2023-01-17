import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
median = reviews.points.median()
print(median)
check_q1(median)
countries = reviews.country.unique()
print(countries)
check_q2(countries)
mo = reviews.country.value_counts()
print(mo)
check_q3(mo)
median = reviews.price.median()
m = reviews.price.map(lambda x: x - median)
check_q4(m)
bestIndex = (reviews.points / reviews.price).idxmax()
best = reviews.loc[bestIndex].title
print(best)
check_q5(best)
def getCounts(column):
    return reviews.description.map(lambda s: column in s).value_counts()
t = getCounts("tropical")
f = getCounts("fruity")
result = pd.Series([t[True], f[True]], index=["tropical", "fruity"])
print(result)
check_q6(result)

filtered = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
conc = filtered.apply(lambda s: s.country + " - " + s.variety, axis="columns")
check_q7(conc.value_counts())
import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
reviews.points.median()
reviews.country.unique()
reviews.country.value_counts()
reviews_price_median = reviews.price.median()
reviews.price.map(lambda p: p - reviews_price_median)

reviews.loc[(reviews.points/reviews.price).argmax()].title

tropical = reviews.description.map(lambda t: "tropical" in t).value_counts()
fruity = reviews.description.map(lambda t: "fruity" in t).value_counts()
wine_combine = pd.Series([tropical[True], fruity[True]], index=['tropical', 'fruity'])
print(wine_combine)
ans = reviews.loc[(reviews.country.notnull())&(reviews.variety.notnull())]
ans = ans.apply(lambda s: s.country + "-" + s.variety, axis = 'columns')
ans = ans.value_counts()
print(ans)
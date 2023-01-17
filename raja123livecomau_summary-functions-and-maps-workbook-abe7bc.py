import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
np.median(reviews.points)
answer_q1()
reviews.country.unique()
answer_q2()
reviews.country.value_counts()
answer_q3()
median_price = reviews.price.median()
reviews.price.map(lambda x: x - median_price)
pd.DataFrame(reviews.price.map(lambda x: x - median_price))
reviews.loc[(reviews.points / reviews.price).argmax()].title

tropical_wine = reviews.description.map(lambda r: "tropical" in r).value_counts()
fruity_wine = reviews.description.map(lambda r: "fruity" in r).value_counts()
pd.Series([tropical_wine[True], fruity_wine[True]], index=['tropical', 'fruity'])

ans = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
ans = ans.apply(lambda x: x.country + " - " + x.variety, axis='columns')
ans.value_counts()
import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
check_q1(reviews.points.median())
check_q2(reviews.country.unique())
check_q3(reviews.country.value_counts())
review_points_median = reviews.price.median()
check_q4(reviews.price.map(lambda p: p - review_points_median))
maximo=(reviews.points / reviews.price).idxmax()
check_q5(reviews.title.loc[maximo])
tropical = reviews.description.map(lambda d: "tropical" in d).value_counts()
fruity = reviews.description.map(lambda d: "fruity" in d).value_counts()
result = pd.Series([tropical[True], fruity[True]], index=['tropical', 'fruity'])
check_q6(result)
result = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
ans = result.apply(lambda row: row.country + " - " + row.variety, axis='columns')
check_q7(ans.value_counts())
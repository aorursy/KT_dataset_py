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
median = reviews.price.median()
check_q4(reviews.price.map(lambda a: a-median))
check_q6(reviews.loc[(reviews.points / reviews.price).idxmax()].title)
check_q7(pd.Series([reviews.description.map(lambda r: w in r).value_counts()[True] for w in words], index=words))
check_q8(reviews.loc[reviews.country.notnull() & reviews.variety.notnull()].apply(lambda x: x.country + " - " + x.variety, axis='columns').value_counts())
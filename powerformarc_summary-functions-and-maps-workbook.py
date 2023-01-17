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
median_price = reviews.price.median()
check_q4(reviews.price.map(lambda price: price - median_price))
p2p_ratio = reviews.points / reviews.price
reviews.iloc[p2p_ratio.idxmax()].title
tropical_count = reviews.description.str.contains("tropical").sum()
fruity_count = reviews.description.str.contains("fruity").sum()
series = pd.Series([tropical_count, fruity_count], index=['tropical', 'fruity'])
series
df = reviews.filter(items=['country', 'variety']).dropna()
(df.country + ' - ' + df.variety).value_counts()
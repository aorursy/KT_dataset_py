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
check_q4(reviews.price.map(lambda p: p - median_price))
check_q5(reviews.price.apply(lambda p: p - median_price))
check_q6(reviews.iloc[(reviews.points/reviews.price).idxmax()].title)
fruity_wine = reviews.description.map(lambda x: 'fruity' in x).value_counts()
tropical_wine = reviews.description.map(lambda x: 'tropical' in x).value_counts()
check_q7(pd.Series([tropical_wine[True], fruity_wine[True]], index=['tropical', 'fruity']))
reduced_data = reviews[reviews.country.notnull() & reviews.variety.notnull()]
reduced_data = reduced_data.apply(lambda x:x.country + ' - ' + x.variety, axis='columns')
check_q8(reduced_data.value_counts())

import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
reviews.points.median()
check_q1(reviews.points.median())
reviews.country.unique()
check_q2(reviews.country.unique())
reviews.country.value_counts()
check_q3(reviews.country.value_counts())
reviews.price - reviews.price.median()
check_q4(reviews.price - reviews.price.median())
reviews.iloc[(reviews.points/reviews.price).idxmax()].title
tropical = reviews.description.map(lambda x: 'tropical' in x)
fruity = reviews.description.map(lambda x: 'fruity' in x)
pd.Series([sum(tropical), sum(fruity)], index=['tropical', 'fruity'])
check_q7(pd.Series([sum(tropical), sum(fruity)], index=['tropical', 'fruity']))
non_null = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
country_variety = (non_null.country + ' - ' + non_null.variety).value_counts()
country_variety
check_q8(country_variety)
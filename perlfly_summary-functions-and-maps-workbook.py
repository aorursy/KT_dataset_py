import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
check_q1(reviews['points'].median())
# Your code here
check_q2(reviews.country.unique())
# Your code here
check_q3(reviews.country.value_counts())
# Your code here
price_median = reviews.price.median()
check_q4(reviews.price.map(lambda p: p - price_median))
# Your code here

price_median = reviews.price.median()
check_q4(reviews['price'].apply(lambda p: p - price_median))
# Your code here
check_q5(reviews.loc[(reviews.points/reviews.price).idxmax()].title)
# Your code here
tropical_wine = (reviews.description.str.find('tropical') >= 0).sum()
fruity_wine = (reviews.description.str.find('fruity') >= 0).sum()
check_q6(pd.Series([tropical_wine, fruity_wine], index=['tropical', 'fruity']))
# Your code here
df = reviews[['country', 'variety']].dropna()
check_q7(df.apply(lambda r: r.country + " - " + r.variety, axis=1).value_counts())
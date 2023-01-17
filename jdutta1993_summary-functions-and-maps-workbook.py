import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
reviews.describe()
check_q1(pd.DataFrame())
reviews.head()
check_q1(reviews.points.median())
check_q2(reviews.country)
check_q3(reviews.country.value_counts())
median_price = reviews.price.median()
new_price = reviews.price.map(lambda p: p - median_price)
check_q4(new_price)
PointsRatio = (reviews.points/reviews.price).idxmax
show = reviews.iloc[PointsRatio]
print(show.title)
check_q6(show.title)
tropical = 'tropical'
fruity = 'fruity'
check_tropic = reviews.description.map(lambda p: tropical in p).sum()
check_fruit = reviews.description.map(lambda p: fruity in p).sum()
taste_count = pd.Series(data = [check_tropic, check_fruit], index = ['tropical', 'fruity'])
print(taste_count)
check_q7(taste_count)
data = reviews[['country','variety']]
data.dropna()
country_index = data['country']+ "-" + data['variety']
check_q8(country_index.value_counts())

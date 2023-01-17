import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
answer_q2()
reviews.head()
q1 = reviews.points.median()
reviews.country.unique()
reviews.country.value_counts()
price_mean = reviews.price.median()

reviews.price.map(lambda p: p - price_mean)
reviews['best_bargain'] = reviews.points/reviews.price
reviews.iloc[reviews.best_bargain.idxmax()]
fruity_count = 0
tropical_count = 0

for el in reviews.description:
    if 'fruity' in el:
        fruity_count += 1
    elif 'tropical' in el:
        tropical_count += 1
    else:
        next

fruity_count, tropical_count
country_list = reviews.country.unique()
variety_list = reviews.variety.unique()
country_and_variety = {}

for country in country_list:
    for variety in variety_list:
        country_and_variety["{} {}".format(country, variety)] = 0

for index, row in reviews.iterrows():
    country = row.country
    variety = row.variety
    country_and_variety["{} {}".format(country, variety)] += 1

    
answer = pd.Series(country_and_variety, index=country_and_variety.keys())
answer.head()

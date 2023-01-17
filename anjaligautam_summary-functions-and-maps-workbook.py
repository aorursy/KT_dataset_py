import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
median_points = reviews.points.median()
print(median_points)
check_q1(median_points)
# Your code here
unique_countries = pd.Series(reviews.country).unique()
check_q2(unique_countries)

# Your code here
Value_counts = reviews.country.value_counts()
check_q3(Value_counts)
# Your code here
median_price = reviews.price.median()
reviews['price'].map(lambda x: x-median_price) 
check_q4(reviews.price)
# Your code here
expected = reviews.loc[(reviews.points/reviews.price).argmax()].title
check_q5(expected)
# Your code here
reviews['tropical_count'] = reviews['description'].str.contains('tropical')
reviews['Fruity_count'] = reviews['description'].str.contains('fruity')
Series_6 = pd.Series([reviews.tropical_count.sum(),reviews.Fruity_count.sum()],index=['Tropical','Fruity'])
print(Series_6)
check_q6(Series_6)
Reviews_2 = reviews.loc[(reviews.country.isnull() == False) & (reviews.variety.isnull() == False)]
Series_8 = Reviews_2.apply(lambda x : x.country + "-" + x.variety)
Series_8.value_counts()
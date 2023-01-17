import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
reviews.points.median()
# Your code here
reviews.country.unique()
# Your code here
reviews.country.value_counts()
# Your code here
median_price = reviews.price.median()
reviews.price.map(lambda p: p - median_price)
# Your code here
ptp_series = reviews.points / reviews.price
best_bargain = reviews.iloc[ptp_series.argmax()]
best_bargain.title

# Your code here
tropical = reviews.description.map(lambda p: 'tropical' in p)
fruity = reviews.description.map(lambda p: 'fruity' in p)
check_q7(pd.Series([tropical.sum(), fruity.sum()], index=['tropical', 'fruity']))

# Your code here
country_var = reviews.loc[:,['country', 'variety']].dropna()
wines_and_country = country_var.country.map(lambda p: p + " - ") + country_var.variety
check_q8(wines_and_country.value_counts())
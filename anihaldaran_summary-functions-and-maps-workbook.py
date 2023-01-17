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
ReviewsPriceMedian=reviews.price.median()
reviews.price.map(lambda p: p - ReviewsPriceMedian)
# Checks how many times each taster name come up in the dataframe
reviews.taster_name.value_counts()


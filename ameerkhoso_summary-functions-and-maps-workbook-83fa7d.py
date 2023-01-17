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
max_countries = reviews.country.value_counts()
print(max_countries)
# Your code here
remap_price = reviews.price.median()
reviews.price.map(lambda p : p - remap_price)
# Your code here
# Your code here
# Your code here
# Your code here
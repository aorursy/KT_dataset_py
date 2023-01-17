import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
reviews.points.median()
reviews.country.unique()
reviews.country.value_counts()
z=reviews.price.median()
reviews.price.map(lambda p:p-z)
reviews.loc[(reviews.points / reviews.price).idxmax()]
# Your code here
df=reviews.dropna(subset=['country','variety'])
(df.country + " - " + df.region_1).value_counts()
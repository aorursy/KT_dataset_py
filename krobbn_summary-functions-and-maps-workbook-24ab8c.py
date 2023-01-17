import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
df = reviews
df.points.median()
df.head()
df.country.unique()
df.country.value_counts()
df_mean = df.points.mean()
df.points - df_mean
df_pricepoint = df[['points', 'price','title']]
df1 = df_pricepoint.points
df2 = df_pricepoint.price
df3 = df1/df2


# Your code here
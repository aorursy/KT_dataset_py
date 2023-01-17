import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
df = reviews.points.median()

check_q1(df)
df = reviews.country.unique()

check_q2(df)
# Your code here
df = reviews.country.value_counts()

check_q3(df)
# Your code here
median_price = reviews.price.median()
df = reviews.price.map(lambda x: x - median_price)

check_q4(df)
# Your code here
name = reviews.loc[(reviews.points / reviews.price).idxmax()].title

check_q6(name)
# Your code here
tropical_wine = reviews.description.map(lambda x: 'tropical' in x).value_counts()
fruity_wine = reviews.description.map(lambda x: 'fruity' in x).value_counts()

df = pd.Series([tropical_wine[True], fruity_wine[True]], index=['tropical', 'fruity'])

check_q7(df)
# Your code here
df = reviews[['country', 'variety']].dropna()
country_variety = df.country + " - " + df.variety
df = country_variety.value_counts()

check_q8(df)
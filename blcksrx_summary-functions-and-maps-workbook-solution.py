import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
check_q1(reviews.points.median())
check_q2(reviews.country.unique())
check_q3(reviews.country.value_counts())
medain = reviews.price.median()
check_q4(reviews.price.map(lambda x: x - medain))
check_q6(reviews.loc[(reviews.points / reviews.price).idxmax()].title)
tropical = reviews.description.map(lambda x: 'tropical' in x)
fruity = reviews.description.map(lambda x: 'fruity' in x)
check_q7(pd.Series({ "tropical": tropical.value_counts()[True], "fruity": fruity.value_counts()[True] }))
df = reviews[(reviews.country.notnull()) & (reviews.variety.notnull())]
check_q8((df.country + " - " + df.variety).value_counts())
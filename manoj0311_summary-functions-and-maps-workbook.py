import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
d = reviews.points.median()
print(check_q1(d))
d = reviews.country
print(check_q2(d.real))
d = reviews.country
print(check_q3(d.value_counts()))
d = reviews.price.median()
print(check_q4(reviews.price.map(lambda p: p - d)))
d = (reviews.points/reviews.price).idxmax()
dd = reviews.iloc[d].title
print(check_q6(dd))
d = reviews.description
trop = reviews.description.map(lambda p : 'tropical' in p)
fruit = reviews.description.map(lambda p : 'fruity' in p)
print(trop.value_counts())
tr=trop.value_counts()[1]
fr=fruit.value_counts()[1]
print(check_q7(pd.Series([tr,fr], index=['tropical','fruity'])))
data = pd.DataFrame(reviews, columns=['country','variety'])
df= data.country +' - '+ data.variety
d = pd.Series(df.value_counts())
print(check_q8(d))
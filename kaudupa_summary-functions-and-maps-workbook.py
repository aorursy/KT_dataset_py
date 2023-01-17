%config IPCompleter.greedy=True
import scipy as sp
import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
reviews.head()
reviews.points.median()
reviews.country.unique()
reviews['country'].value_counts().head()
mean_price=reviews.price.mean()
reviews.price.map(lambda p:mean_price-p)
reviews.loc[(reviews.points/reviews.price).idxmax(),['winery']]
tropical = reviews.description.map(lambda p: True if "tropical" in p.lower() else False)
fruity=reviews.description.map(lambda p: True if "fruity" in p.lower() else False)
for element in tropical:
    if element:
        count_tropical+=1
for element in fruity:
    if element:
        count_fruity+=1
print(count_tropical,count_fruity)
reviews.country + " - " + reviews.region_1
import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(20)
reviews.head()
reviews.points.median()
print(reviews.country.unique())
reviews["country"].value_counts()
reviews["price"] = reviews.price - reviews.price.median()
reviews.head()
reviews["ratio"] = reviews["points"]/reviews["price"]
reviews.loc[reviews.ratio.idxmax].title
tropical_count = reviews["description"].map(lambda x: "tropical" in x.split(" ")).value_counts()
fruity_count = reviews["description"].map(lambda x: "fruity" in x.split(" ")).value_counts()
pd.Series([tropical_count[True], fruity_count[True]], index = ["tropical", "fruity"])
complete_reviews = reviews.loc[reviews["country"].notnull() & reviews["variety"].notnull()]
complete_reviews = complete_reviews["country"] + "-" + complete_reviews["variety"]
complete_reviews.value_counts()
import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
# Your code here
# Your code here
# Your code here

def price_map(x): return x-85
price_map_list=list(map(price_map,reviews.price))
reviews.price=price_map_list
reviews.price
# Your code here
reviews.ppr=[]
def ppr(x):
    for i in reviews.points:
        ppr=i/x
    return ppr
ppr_map_list=list(map(ppr,reviews.price))
reviews.ppr=ppr_map_list
reviews.ppr
# Your code here

# Your code here
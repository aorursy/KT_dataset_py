import pandas as pd
import seaborn as sns

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
# Your code here
reviews.description
# Your code here
reviews.description[0]
# Your code here
reviews.iloc[0]
# Your code here
reviews.iloc[:10,1]
# Your code here
reviews.iloc[[1,2,3,4,5,8]]
# Your code here
reviews.iloc[[0,1,10,100], [0,5,6,7]]
# Your code here
reviews.loc[:99]
# Your code here
reviews.loc[reviews.country == 'Italy']
# Your code here
reviews.loc[reviews.region_2.notnull()]
# Your code here
reviews.points
# Your code here
reviews.loc[:999, 'points']
# Your code here
reviews.iloc[-1000: , 3]
# Your code here
reviews.loc[reviews.country == 'Italy', 'points']
# Your code here
reviews.loc[(reviews.country.isin(['France', 'Italy'])) & (reviews.points >= 90), 'country']
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
(reviews['description'])
# Your code here
d = reviews['description']
d[0]
# Your code here
rdf = reviews.iloc[0]
print(rdf)
# Your code here
reviews.iloc[1:10,1]
# Your code here
reviews.iloc[[1,2,3,4,5,8]]
# Your code here
reviews.loc[[0, 1, 10, 100], ['country', 'province', 'region_1', 'region_2']]
# Your code here
reviews.loc[0:100,['country','variety']]
# Your code here
reviews[reviews['country']=='Italy'].count()
reviews.count() # 129908l
#print(review_italy_count,review_count)
# Your code here
reviews[reviews['region_2'].notnull()]

# Your code here
reviews['points']
# Your code here
reviews.loc[:1000,'points']
# Your code here
reviews.iloc[-1000:,3]
# Your code here
reviews[reviews.country == "Italy"].points
# Your code here
reviews[reviews.country.isin(['Italy','France']) & (reviews.points >=90)]
import pandas as pd
import seaborn as sns

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
answer_q1()
check_q1(reviews.description)
reviews.description
check_q9(reviews.description.loc[1])
# Your code here
reviews.description.loc[1]
# Your code here
reviews.description.loc[1]
reviews.description[1:10]
reviews.loc[[1,2,3,5,8]]
# Your code here
reviews.loc[[0,1,10,100],['country','province','region_1','region_2']]

# Your code here
reviews.loc[:100,["country",'variety']]
# Your code here
reviews.loc[reviews.country  == "Italy"]
# Your code here
reviews.loc[reviews.region_2.isnull()]
# Your code here
reviews.points
# Your code here

reviews['points'][:1000]
# Your code here
reviews['points'][:-1000]
# Your code here
reviews[reviews.country == "Italy"]['points']
# Your code here

reviews[reviews.points>=90]['country']


import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
# Your code here
reviews.description
#or also: reviews['description']
#answer_q1()
# Your code here
reviews.loc[0,'description']
# Your code here
reviews.iloc[0]
# Your code here
#reviews.loc[:10,'description']
#ds = pd.Series(reviews.loc[:10,'description'])
#ds
reviews.iloc[0:10, 1]
# Your code here
reviews.iloc[[1,2,3,5,8]]
# Your code here
reviews.iloc[[0,1,10,100],[0,5,6,7]]
# Your code here
#reviews.loc[:99,['country','variety']]
reviews.iloc[:100,[0,11]]

# Your code here
reviews.loc[reviews.country=='Italy']
# Your code here
reviews.loc[reviews.region_2.notnull()]
# Your code here
check_q10(reviews.points)

# Your code here
reviews.loc[:999,'points']
# Your code here
reviews.loc[:-999,'points']
# Your code here
reviews.loc[reviews.country=='Italy','points']
#reviews[reviews.country=='Italy'].points
#answer_q13()
# Your code here
reviews[(reviews.points>=90) & (reviews.country.isin(['France','Italy']))].country

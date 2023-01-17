import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
# Your code here
reviews['description']
# Your code here
reviews.description[0]
# Your code here
reviews.iloc[0]
answer_q3()
# Your code here
reviews.loc[0:9, 'description']
# Your code here
reviews.iloc[[1,2,3,5,8]]
# Your code here
reviews.loc[[0,1,10,100], ['country', 'province', 'region_1', 'region_2']]
# Your code here
reviews.loc[0:99, ['country', 'variety']]
# Your code here
reviews[reviews.country == 'Italy']
# Your code here
reviews[reviews.region_2.notnull()]
# Your code here
reviews['points']
# Your code here
reviews.points[0:1000]
#answer_q11()
# Your code here
reviews.points[-1000:]
# Your code here
#reviews['points'][reviews.country == 'Italy']
reviews[reviews.country == 'Italy'].points
# Your code here
reviews[reviews.country.isin(['France', 'Italy']) & (reviews.points >= 90)]

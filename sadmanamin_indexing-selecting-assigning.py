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
reviews['description'].loc[0]
# Your code here
reviews.loc[0]
# Your code here
reviews['description'].head(10)
# Your code here
reviews.loc[[1,2,3,5,8]]

# Your code here
reviews[['country','province','region_1','region_2']].loc[[0,1,10,100]]
# Your code here
reviews[['country','variety']][0:101]
# Your code here
reviews[reviews.country == 'Italy']
# Your code here
reviews.loc[reviews['region_2'].notnull()]
# Your code here
reviews.points
# Your code here
reviews.points[0:1001]
# Your code here
reviews.points.tail(1000)
# Your code here
reviews.points[reviews.country == 'Italy']
# Your code here
reviews.country[reviews.points >=90]
import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
reviews.info()
reviews['description'].head(5)
reviews['description'].head(1)
reviews.loc[0]
reviews.loc[0:9, 'description']
pd.DataFrame(reviews.loc[[1, 2, 3, 5, 8]])
reviews.loc[[0, 1, 10, 100], ['country', 'province', 'region_1', 'region_2']]
reviews.loc[0:100, ['country', 'variety']]
reviews[reviews['country'] == 'Italy']
reviews['region_2'] == 'NaN'
reviews['points']
reviews.loc[0:1000, 'points']
reviews['points'].tail(1000)
# Your code here
# Your code here
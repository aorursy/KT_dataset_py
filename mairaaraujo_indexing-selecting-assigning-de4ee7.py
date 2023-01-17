import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
reviews['description']
reviews['description'][0]
reviews.iloc[0]
pd.Series(reviews.loc[:9, 'description'])
pd.DataFrame(reviews.iloc[[1,2,3,5,8]])
pd.DataFrame(reviews.loc[[0,1,10,100], ['country', 'province', 'region_1', 'region_2']])
pd.DataFrame(reviews.loc[:99, ['country', 'variety']])
reviews.country == 'Italy'
reviews.region_2 != 'NaN'
reviews.points
reviews.loc[:999, 'points']
reviews.iloc[-1000:]['points']
reviews.loc[reviews.country == 'Italy']['points']
reviews.loc[(reviews.country.isin(['Italy', 'France'])) & (reviews.points >= 90)]['country']
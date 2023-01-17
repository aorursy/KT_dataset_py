import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
reviews.description
reviews.description[0]
reviews.loc[0,:]
reviews.loc[:9,['description']]
reviews.iloc[['1','2','3','5','8']]
reviews.loc[[0,1,10,100],['country','province','region_1','region_2']]
reviews.loc[:99,['country','variety']]
reviews.country =='Italy'
reviews.region_2 !='NaN'
reviews.points
reviews.points[:1000]
reviews.points[-1000:]
reviews.points[reviews.country=='Italy']
reviews.country[(reviews.country.isin(['Italy','France'])) & (reviews.points >=90)]
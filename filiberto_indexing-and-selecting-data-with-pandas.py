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
reviews.description[:10]
reviews.iloc[[1,2,3,5,8],:]   # We can do it with a list!!
reviews.loc[[0,1,10,100],['country','province','region_1','region_2']]
reviews.loc[:99,['country','variety']]
reviews.country == 'Italy'
reviews.loc[reviews.region_2.notnull()]
reviews.points
reviews.points[:1000]
reviews.points[-1000:]   
reviews.loc[reviews.country == 'Italy'] 
reviews.country.loc [((reviews['country'] == 'Italy') |  (reviews['country'] == 'France')) & (reviews['points'] >= 90)]
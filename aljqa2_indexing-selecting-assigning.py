import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
reviews['description']
reviews['description'].iloc[0]
#reviews.iloc[0]
#reviews.loc[reviews.index[0]]
reviews.head(1)
ser = reviews.loc[0:9, 'description']
df = pd.DataFrame(reviews.iloc[[1,2,3,5,8],:])
df
#df = pd.DataFrame(reviews.iloc[[0,1,10,100],['country', 'province', 'region_1', 'region_2']])
#df
reviews.loc[[0,1,10,100],['country', 'province', 'region_1', 'region_2']]
reviews.loc[0:99, ['country', 'variety']]
reviews.loc[reviews.country == 'Italy']
reviews.loc[reviews.region_2.notnull()]
reviews['points']
reviews.loc[0:999, ['points']]
#reviews.loc[-999:0, ['points']]
#reviews['points'].loc[-999
reviews.iloc[-1000:, 3]
#answer_q12()
#reviews['points'].loc[(reviews.country == 'Italy')]
#reviews[reviews.country == 'Italy'].points
reviews[reviews.country == "Italy"].points
#answer_q13()
reviews[reviews.country.isin(['Italy', 'France']) & (reviews.points >=90)].country
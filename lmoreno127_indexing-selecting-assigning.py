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
reviews['description'][:10]
reviews.iloc[[1,2,3,5,8]]
reviews.loc[[0,1,10,100],['country','province','region_1','region_2']]
reviews.iloc[:100][['country','variety']]
reviews.loc[reviews.country == 'Italy']
reviews.loc[reviews.region_2.notnull()]
check_q10(reviews['points'])
check_q11(reviews.iloc[:1000]['points'])
check_q12(reviews.iloc[-1000:]['points'])
reviews[reviews.country=='Italy'].points
reviews.loc[reviews.country.isin(['Italy', 'France']) & (reviews.points >= 90) ,'country']
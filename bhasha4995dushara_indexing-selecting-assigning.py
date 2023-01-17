import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head(10)
check_q1(pd.DataFrame())
reviews['description']
reviews.loc[0,'description']  # or
#reviews['description'][0]
#reviews.iloc[0] # or
reviews.loc[0,:]
f = pd.Series(reviews['description'].head(10))
f
reviews.iloc[[1,2,3,5,8]]
reviews[['country','province','region_1','region_2']].iloc[[0,1,10,100]]
#reviews[['country','variety']].iloc[0:100]  # or
reviews[['country','variety']].loc[0:99]
reviews[reviews['country'] == 'Italy']['winery']
reviews[reviews['region_2'].notnull()]['winery']
reviews['points']
reviews.iloc[:999]['points']
#reviews['points'].tail(1000)   # or
reviews.iloc[-1000:]['points']
reviews.loc[reviews.country == 'Italy']['points']
reviews.loc[(reviews['points'] >= 90) & (reviews['country'].isin(['Italy','France']))]['country']
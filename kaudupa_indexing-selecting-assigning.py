%config IPCompleter.greedy=True
import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
reviews.description
reviews.description[0]
reviews.loc[0]
ps=pd.Series(reviews.description.head(10))
ps
reviews.iloc[[1,2,3,5,8]]
reviews.loc[[0,1,10,100],['country','province','region_1','region_2']]
reviews.loc[0:100,['country','variety']]
reviews[reviews['country']=='Italy']['winery']
import numpy as np
reviews['winery'][reviews.region_2!=np.NaN]
reviews.points
reviews.loc[0:999,['points','winery']]
reviews.loc[reviews.index[129970:128970:-1],['points']]
# reviews.points[reviews.country=='Italy']
reviews[reviews['country']=='Italy']['points']
# reviews.country[((reviews['country']==('Italy')) | (reviews['country']==('France'))) & (reviews['points']>=90)]
reviews[(reviews['points']>=90)&((reviews['country']=='France') | (reviews['country']=='Italy'))]['country']
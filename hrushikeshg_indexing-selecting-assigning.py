import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
reviews.iloc[:,1:2]

check_q1(pd.DataFrame())
reviews.iloc[0:1,1:2]
check_q2(pd.DataFrame())
reviews.iloc[0:1,:]
reviews.iloc[0:11,1:2]
df1=reviews.iloc[[1,2,3,5,8],:]
print(df1)

reviews.head(1)
reviews.iloc[[0,1,10,100],[0,5,6,7]]
reviews.iloc[0:101,[0,11]]
reviews[reviews.country=='Italy']
reviews[reviews['region_2'].isnull()]
reviews['points']
reviews['points'].head(1001)
reviews['points'].tail(1001)
reviews[reviews['country']=='Italy']['points']
reviews[(reviews.points>=90) & ((reviews.country=='France')| (reviews.country=='Italy'))]
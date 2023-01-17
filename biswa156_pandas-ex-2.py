import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
# Your code here
#reviews.head(2)
#reviews.columns
check_q1(reviews['description'])
#reviews.description
# Your code here
check_q2(reviews.loc[[0],'description'])
# Your code here
#reviews.iloc[[0],[2]]
check_q3(reviews.iloc[0,:])
#reviews.iloc[0,:]
# Your code here
check_q4(pd.Series(reviews.loc[0:9,'description']))
# Your code here
check_q5(reviews.iloc[[1,2,3,5,8],])
# Your code here
check_q6(reviews.loc[[0,1,10,100],['country','province','region_1','region_2']])
# Your code here
check_q7(reviews.loc[0:99,['country','variety']])
# Your code here
check_q8(reviews[reviews.country=='Italy'])

# Your code here
#check_q9(reviews[pd.notna(reviews.region_2)])
check_q9(reviews[pd.notna(reviews.region_2)==1])
# Your code here
check_q10(reviews['points'])
# Your code here
check_q11(reviews.loc[0:999,'points'])
# Your code here
#check_q12(reviews.iloc[-1000:]['points'])
check_q12(reviews.points[-1000:])
# Your code here
check_q13(reviews.loc[reviews.country=='Italy','points'])
# Your code here
#reviews.loc[(reviews.points)>=90 | (reviews.country.any('France','Italy'))]
check_q14(reviews.loc[(reviews['country'].isin(['France','Italy'])) & (reviews['points'] >= 90),'country'])

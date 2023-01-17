import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
# Your code here
reviews.ix[:,'description']
check_q1(reviews.ix[:,'description'])
# Your code here
check_q2(reviews.loc[0,'description'])
# Your code here
check_q3(reviews.loc[0,:])
# Your code here
pd.Series(reviews.description[10:])
# Your code here
reviews.iloc[[1,2,3,5,8]]
# Your code here
reviews.loc[[0,1,10,100],['country','province','region_1','region_2']]

# Your code here
reviews.loc[0:100,['country','variety']]
# Your code here
reviews.loc[reviews.country=='Italy']

# Your code here
Series=reviews.loc[reviews.region_2.notnull()]
check_q9(Series)
# Your code here
reviews.points
# Your code here
check_q11(reviews.points[0:1000])
# Your code here
series=reviews.iloc[-1000:,3]
series
check_q12(series)

# Your code here
series=reviews.points[reviews.country=='Italy']
check_q13(series)
# Your code here
series=reviews.country[reviews.country.isin(['Italy','France'])& (reviews.points>=90)]
series
check_q14(series)

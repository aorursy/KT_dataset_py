import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
# Your code here
reviews.description

answer_q1()
# Your code here
reviews.description[0]

# Your code here
reviews.iloc[0]

# Your code here
reviews.iloc[:10,1]

# Your code here
reviews.iloc[[1,2,3,5,8]]


# Your code here
reviews[['country','province','region_1','region_2']].iloc[[0,1,10,100]]

# Your code here
reviews[['country','variety']].iloc[0:100]

# Your code here
reviews.loc[reviews['country'] =='Italy']

# Your code here
reviews.loc[reviews['region_2'].notnull()]
# Your code here
reviews.loc[reviews['points']]


# Your code here
reviews['points'].iloc[0:1000]
# Your code here
reviews['points'].iloc[-1000:]

# Your code here
reviews[reviews['country'] == 'Italy']['points']

# Your code here
reviews[(reviews['points'] > 89) & reviews['country'].isin(['Italy','France'])]

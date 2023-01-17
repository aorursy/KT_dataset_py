import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
# Your code here
reviews.description
check_q1(reviews.description)
# Your code here
reviews['description'][0]
check_q2(reviews['description'][0])
# Your code here
reviews.iloc[0]
print(reviews.iloc[0])
check_q3(reviews.iloc[0])
# Your code here
srs = reviews['description'].head(10)
check_q4(srs)
# Your code here
reviews.iloc[[1,2,3,5,8]]
check_q5(reviews.iloc[[1,2,3,5,8]])
# Your code here
df_new = reviews.loc[[0,1,10,100], ['country', 'province', 'region_1', 'region_2']]
check_q6(df_new)
# Your code here
res = reviews.loc[0:99][['country', 'variety']]
check_q7(res)
# Your code here
wines_italy = reviews[reviews.country == 'Italy']
check_q8(wines_italy)
# Your code here
regn_2 = reviews.loc[reviews['region_2'].isnull() == False]
check_q9(regn_2)
# Your code here
points = reviews['points']
check_q10(points)
# Your code here
first_1000 = reviews.loc[0:999]['points']
check_q11(first_1000)
# Your code here
# Your code here
# Your code here
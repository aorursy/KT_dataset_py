import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
check_q1(reviews['description'])
check_q2(reviews['description'][0])
check_q3(reviews.iloc[0])
check_q4(reviews['description'][0:10])
print(answer_q5())
print(answer_q6())
print(answer_q7())
check_q8(reviews[reviews.country == 'Italy'])
reviews.loc[reviews.region_2.notnull()]
check_q10(reviews['points'])
# Your code here
# Your code here
# Your code here
# Your code here
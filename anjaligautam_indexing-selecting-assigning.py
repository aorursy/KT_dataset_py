import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
# Your code here
Description_column = reviews['description']
check_q1(Description_column)
# Your code here
x = reviews['description'][0]
print(x)
check_q2(x)
# Your code here
First_rec = reviews.iloc[0]
check_q3(First_rec)
# Your code here
Desc_series = pd.Series(reviews['description'][0:10])
check_q4(Desc_series)
# Your code here
reviews_subset = reviews.iloc[[1,2,3,5,8]]
check_q5(reviews_subset)
# Your code here
Data_frame6 = reviews.loc[[0,1,10,100],['country','province','region_1','region_2']]
check_q6(Data_frame6)
# Your code here
First_100 = reviews.loc[0:100,['country','variety']]
check_q7(First_100)
# Your code here
Italy_wines = reviews[reviews.country=='Italy']
check_q8(Italy_wines)
# Your code here
Wines_9 = reviews[reviews.region_2.isnull() == 0]
check_q9(Wines_9)

# Your code here
points_column = reviews['points']
check_q10(points_column)
# Your code here
points_column_100 = reviews['points'][0:1001]
check_q11(points_column_100)
# Your code here
last_1000 = reviews['points'][-1000:]
check_q12(last_1000)
# Your code here
points_Italy = reviews[reviews.country == 'Italy']['points']
check_q13(points_Italy)
# Your code here
Dataframe_14 = reviews[reviews.country.isin(['Italy','France']) & (reviews.points >= 90)].country
check_q14(Dataframe_14)
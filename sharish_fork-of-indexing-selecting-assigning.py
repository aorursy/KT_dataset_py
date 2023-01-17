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
check_q5(reviews.loc[[1,2,3,5,8]])
check_q6(reviews[['country','province','region_1','region_2']].loc[[0,1,10,100]])
check_q7(reviews[['country','variety']].iloc[0:100])
check_q8(reviews[reviews['country']=='Italy'])
check_q9(reviews[reviews['region_2'].notnull()])

check_q10(reviews.points)
check_q11(reviews.points[0:1000])
check_q12(reviews.iloc[-1000:, 3])
check_q13(reviews[reviews['country']=='Italy'].points)
country=reviews[reviews['country'].isin(['France','Italy'])]
check_q14(country[country['points']>=90].country)
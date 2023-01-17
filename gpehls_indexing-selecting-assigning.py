import pandas as pd
import seaborn as sns

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q6(pd.DataFrame())
check_q1(reviews.description)
check_q2(reviews.description[0])
check_q3(reviews.iloc[0])
check_q4(reviews.description.iloc[0:10])
check_q5(reviews.iloc[[1,2,3,5,8]])
check_q6(reviews.loc[[0,1,10,100],['country','province','region_1','region_2']])
check_q7(reviews.loc[0:100,['country','variety']])
check_q8(reviews.loc[reviews.country=='Italy'])
check_q9(reviews.loc[reviews.region_2.notnull()])
check_q10(reviews.points)
check_q11(reviews.points.loc[0:999])
check_q12(reviews.points.iloc[len(reviews)-1000:len(reviews)])
check_q13(reviews.points.loc[reviews.country=='Italy'])
check_q14(reviews[reviews.country.isin(["Italy","France"]) & (reviews.points>=90)].country)
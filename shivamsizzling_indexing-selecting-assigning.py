import pandas as pd
import seaborn as sns

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
des_col = reviews["description"]
check_q1(des_col)
val1 = des_col[0]
check_q2(val1)
col_1 = reviews.iloc[0,:]
check_q3(col_1)
des_ten = des_col[:10]
type(des_ten)
check_q4(des_ten)
q5_records = reviews.iloc[[1,2,3,5,8],:]
check_q5(q5_records)
reviews.columns
q6_records = reviews.loc[[0,1,10,100],["country","province","region_1","region_2"]]
#q6_records = reviews.iloc[[0,1,10,100],[0,5, 6,7]]
check_q6(q6_records)
q7_records = reviews[["country","variety"]]
q7_records1 = q7_records.iloc[:101,:]
q7_records1
check_q7(q7_records1)
#answer_q7()
q8_wines = reviews[reviews.country == "Italy"]
q8_wines
check_q8(q8_wines)
q9_wines = reviews[~reviews.region_2.isna()]
q9_wines
check_q9(q9_wines)
q10_col = reviews["points"]
check_q10(q10_col)
q11_col = reviews.loc[:999, "points"]
q11_col
check_q11(q11_col)
#answer_q11()
q12_col = reviews.iloc[-1000:,3]
#q12_col
check_q12(q12_col)
#answer_q12()


#q10_col = reviews["points"]
#check_q10(q10_col)q8_wines
q13_col = reviews.points[reviews.country == "Italy"]
q13_col
check_q13(q13_col)
q14_col = reviews.country[((reviews.country == "Italy") | (reviews.country == "France")) & (reviews.points >=90) ]
q14_col
#q13_col
check_q14(q14_col)
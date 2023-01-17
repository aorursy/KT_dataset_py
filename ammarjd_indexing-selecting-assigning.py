import pandas as pd
import seaborn as sns

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
desc = reviews['description']
check_q1(desc)
val = desc[0]
check_q2(val)
row = reviews.iloc[0,:]
check_q3(row)
vals_desc = reviews.description[:10]
check_q4(vals_desc)
sel = reviews.iloc[[1, 2, 3, 5, 8]]
check_q5(sel)
sliced = reviews.loc[[0, 1, 10, 100], ['country', 'province', 'region_1', 'region_2']]
check_q6(sliced)
sliced_2 = reviews.loc[0:100, ['country', 'variety']]
check_q7(sliced_2)
wines_italy = reviews.loc[reviews.country == 'Italy']
check_q8(wines_italy)
reg_nonnan = reviews.loc[reviews.region_2.notnull()]
check_q9(reg_nonnan)
pts = reviews.points
check_q10(pts)
pts_1000 = reviews.loc[:1000, 'points']
check_q11(pts_1000)
pts_last_1000 = reviews.iloc[-1000:, 3]
check_q12(pts_last_1000)
pts_ita = reviews.points[reviews.country == 'Italy']
check_q13(pts_ita)
avg_data = reviews[reviews.country.isin(['Italy', 'France']) & (reviews.points >= 90)].country
check_q14(avg_data)

import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
desc = reviews.description
check_q1(desc)
desc0 = desc[0]
check_q2(desc0)
first = reviews.iloc[0]
check_q3(first)
desc = reviews.iloc[0:10, 1]
check_q4(desc)
records = reviews.iloc[[1, 2, 3, 5, 8]]
check_q5(records)
records = reviews.iloc[[0, 1, 10, 100]].loc[:, ['country', 'province', 'region_1', 'region_2']]
check_q6(records)
records = reviews.loc[:99][['country', 'variety']]
check_q7(records)
records = reviews.loc[reviews.country == "Italy", :]
check_q8(records)
records = reviews.loc[reviews.region_2.notnull()]
check_q9(records)
records = reviews.points
check_q10(records)
records = reviews.loc[:999, 'points']
check_q11(records)
records = reviews.iloc[-1000:, 3]
check_q12(records)
records = reviews.loc[reviews.country == "Italy", "points"]
check_q13(records)
records = reviews.loc[reviews.country.isin(["Italy", "France"]) & (reviews.points >= 90)].country
check_q14(records)
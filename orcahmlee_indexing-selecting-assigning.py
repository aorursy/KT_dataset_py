import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
col = reviews['description']
col.head()
check_q1(col)
data = reviews['description'][0]
check_q2(data)
row = reviews.loc[0, :]
check_q3(row)
data = pd.Series(reviews['description'].loc[0:9]) # 笨方法
data = reviews.loc[0:9, 'description'] # loc[] <= 可用 column name
data = reviews.iloc[0:10,1] # iloc[] <= index
check_q4(data)
data = reviews.loc[[1, 2, 3, 5, 8 ],:]
check_q5(data)
data = reviews.loc[[0,1,10,100] , ['country', 'province', 'region_1', 'region_2']]
check_q6(data)
data = reviews.loc[0:100, ['country', 'variety']]
check_q7(data)
data = reviews[reviews['country'] == 'Italy']
check_q8(data)
data = reviews[reviews['region_2'].notnull()]
check_q9(data)
check_q10(reviews.points)
check_q11(reviews.points[0:1000])
check_q11(reviews.points[:-1000])
data = reviews[reviews.country=='Italy']['points']
check_q13(data)
data = reviews[reviews.country.isin(['Italy', 'France']) & (reviews.points >= 90)]['country']
check_q14(data)
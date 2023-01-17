import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
# Your code here
df=reviews.description
df

print(check_q1(df))
# Your code here
df2=reviews.description.head(1)
df2
# Your code here
df3=reviews.iloc[0]
df3
# Your code here
df4 = reviews.description.iloc[0:10]
df4
# Your code here
df5 = reviews.iloc[ [1,2,3,5,8]]
df5
# Your code here
df6 = reviews[['country','province','region_1','region_2']].iloc[[0,1,10,100]]
df6
# Your code here
df7= reviews.iloc[0:101][['country','variety']]
df7
# Your code here
df8=reviews[reviews.country=='Italy']
df8
# Your code here
import numpy as np
df9=reviews[reviews.region_2.notnull()]
df9
# Your code here
df10 = reviews.points
# Your code here
df11= reviews.points[0:1000]
# Your code here
df12 = reviews.tail(1000)['points']
# Your code here
df13= reviews[reviews['country']=='Italy']['points']
# Your code here
df14=reviews[reviews['country'].isin(['Italy','France'])][reviews.points>=90].country
df14
check_q14(df14)
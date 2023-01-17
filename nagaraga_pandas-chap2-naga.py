import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
reviews['description']
check_q1(reviews['description'])
reviews.description[0]
# (reviews['description']) code here
reviews.iloc[:1,:]
#reviews.description[0:10]
reviews.loc[0:9, 'description']
#check_q4(reviews.description[0:10])
reviews.loc[[1,2,3,5,8],]
check_q5(reviews.loc[[1,2,3,5,8],])
reviews.loc[[0, 1, 10, 100], ['country', 'province', 'region_1', 'region_2']]
#check_q5(reviews.loc[[1,2,3,5,8],])
reviews.loc[0:99, ['country', 'variety']]
check_q7(reviews.loc[0:99, ['country', 'variety']])
reviews[reviews.country == 'Italy']
check_q8(reviews[reviews.country == 'Italy'])
#reviews[reviews.region_2 is not 'NaN']
reviews[reviews['region_2'].notnull()]
check_q9(reviews[reviews['region_2'].notnull()])
reviews['points']
check_q10(reviews['points'])
reviews['points'].head(100)
check_q11(reviews['points'].head(100))
reviews['points'].tail(1000)
reviews['points'][reviews['country'] == 'Italy']
filter_list = ['Italy', 'France']

reviews['country'][(reviews.country.isin(filter_list)) & (reviews.points >= 90) ]
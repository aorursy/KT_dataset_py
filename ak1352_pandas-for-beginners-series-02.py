import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.indexing_selecting_and_assigning import *
print("Setup complete.")
reviews.head()
# Your code here
desc = reviews['description']

q1.check()
# Uncomment the line below to see a solution
type(desc)
#q1.solution()
first_description = desc[0]

q2.check()
first_description
#q2.solution()
first_row = reviews.iloc[0] #retrieve first row from dataframe 'reviews'

q3.check()
first_row
#q3.solution()
first_descriptions = reviews.loc[0:9,'description']
# second slution
# first_descriptions = reviews.loc[0:10,1]

q4.check()
first_descriptions
#q4.solution()
sample_reviews = reviews.iloc[[1,2,3,5,8],:]

q5.check()
sample_reviews
#q5.solution()
df = reviews.loc[[0,1,10,100],['country' , 'province' , 'region_1' , 'region_2']] 

q6.check()
df
#q6.solution()
df = reviews.loc[0:99,['country','variety']]

q7.check()
df
#q7.solution()
#solution 01
italian_wines = reviews.loc[reviews.country == 'Italy']
#solution 02 using 'isin' method
#reviews.loc[reviews.country.isin(['Italy'])]

q8.check()
#q8.solution()
top_oceania_wines = reviews.loc[(reviews.country.isin(['Australia' , 'New Zealand'])) & (reviews.points >=95)]

q9.check()
top_oceania_wines
#q9.solution()
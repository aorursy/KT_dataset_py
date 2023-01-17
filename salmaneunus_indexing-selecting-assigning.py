import pandas as pd



reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

pd.set_option("display.max_rows", 5)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.indexing_selecting_and_assigning import *

print("Setup complete.")
reviews.head()
# Your code here

desc = reviews.description



# Check your answer

q1.check()
first_description = reviews['description'][0]



# Check your answer

q2.check()

first_description
first_row = reviews.iloc[0]



# Check your answer

q3.check()

first_row
first_descriptions = reviews.iloc[:10,1]



# Check your answer

q4.check()

first_descriptions
sample_reviews = reviews.iloc[[1,2,3,5,8]]



# Check your answer

q5.check()

sample_reviews
df = reviews.iloc[[0,1,10,100],[0,5,6,7]]



# Check your answer

q6.check()

df
df = reviews.loc[0:99,['country','variety']]



# Check your answer

q7.check()

df
italian_wines = reviews.loc[reviews.country=='Italy']



# Check your answer

q8.check()
top_oceania_wines = reviews.loc[reviews.country.isin(['Australia','New Zealand'])& (reviews.points>=95)] 



# Check your answer

q9.check()

top_oceania_wines
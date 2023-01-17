import pandas as pd



reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

pd.set_option("display.max_rows", 5)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.indexing_selecting_and_assigning import *

print("Setup complete.")
reviews.head()
reviews['description']
# Your code here

desc = reviews['description']



q1.check()
type(desc)
#q1.hint()

#q1.solution()
reviews['description'].iloc[0]
first_description = reviews['description'].iloc[0]



q2.check()

first_description
#q2.hint()

#q2.solution()
reviews.iloc[0]
first_row = reviews.iloc[0]



q3.check()

first_row
#q3.hint()

#q3.solution()
reviews.T.squeeze().iloc[:10]
first_descriptions = reviews.description.iloc[:10]



q4.check()

first_descriptions
#q4.hint()

#q4.solution()
index_list = [1,2,3,5,8]



sample_reviews = reviews[reviews.index.isin(index_list)]



q5.check()

sample_reviews
#q5.hint()

#q5.solution()
index_list = [0,1,10,100]

df = reviews[['country', 'province', 'region_1', 'region_2']][reviews.index.isin(index_list)]



q6.check()

df
#q6.hint()

#q6.solution()
reviews[['country', 'variety']].iloc[:1001]
df = reviews[['country', 'variety']].iloc[:100]



q7.check()

df
#q7.hint()

#q7.solution()
italian_wines = reviews[reviews.country == 'Italy']



q8.check()
#q8.hint()

#q8.solution()
reviews[(reviews.points >=95) & (reviews.country.isin(['Australia', 'New Zealand']))]
top_oceania_wines = reviews[(reviews.points >=95) & (reviews.country.isin(['Australia', 'New Zealand']))]



q9.check()

top_oceania_wines
#q9.hint()

#q9.solution()
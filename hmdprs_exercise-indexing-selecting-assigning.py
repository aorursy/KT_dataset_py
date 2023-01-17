from learntools.core import binder; binder.bind(globals())

from learntools.pandas.indexing_selecting_and_assigning import *

print("Setup complete.")



import pandas as pd

pd.set_option("display.max_rows", 5)



# load data

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
reviews.head()
# your code here

desc = reviews['description']



# check your answer

q1.check()
type(desc)
# q1.hint()

# q1.solution()
first_description = reviews['description'].iloc[0]



# check your answer

q2.check()

first_description
# q2.hint()

# q2.solution()
first_row = reviews.iloc[0]



# check your answer

q3.check()

first_row
# q3.hint()

# q3.solution()
first_descriptions = reviews['description'].iloc[:10]



# check your answer

q4.check()

first_descriptions
# q4.hint()

# q4.solution()
sample_reviews = reviews.iloc[[1, 2, 3, 5, 8]]



# check your answer

q5.check()

sample_reviews
# q5.hint()

# q5.solution()
df = reviews.loc[[0, 1, 10, 100], ['country', 'province', 'region_1', 'region_2']]



# check your answer

q6.check()

df
# q6.hint()

# q6.solution()
df = reviews.loc[0:99, ['country', 'variety']]



# check your answer

q7.check()

df
# q7.hint()

# q7.solution()
italian_wines = reviews[(reviews['country'] == 'Italy')]



# check your answer

q8.check()
# q8.hint()

# q8.solution()
# 1st

# top_oceania_wines = reviews[

#     (reviews['points'] >= 95) & (

#         (reviews['country'] == 'Australia') | (reviews['country'] == 'New Zealand')

#     )

# ]



# 2nd

top_oceania_wines = reviews[

    (reviews['points'] >= 95) & (reviews['country'].isin(['Australia', 'New Zealand']))

]



# check your answer

q9.check()

top_oceania_wines
# q9.hint()

# q9.solution()
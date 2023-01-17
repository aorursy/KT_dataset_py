import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.indexing_selecting_and_assigning import *
print("Setup complete.")
reviews.head()
# Your code here
desc = reviews.description # or reviews['description'] // "desc" is a pandas Series

# Check your answer
q1.check()
print(type(desc))
print(desc)
q2.hint()
first_description = reviews.description[0]

#reviews.description.loc[0]
#reviews.description.iloc[0]

# Check your answer
q2.check()
first_description
first_row = reviews.iloc[0]

# Check your answer
q3.check()
first_row
first_descriptions = reviews.description[0:10]

# reviews.description.iloc[:10]
# reviews.loc[:9, "description"]

# Check your answer
q4.check()
first_descriptions
indices=[1,2,3,5,8]
sample_reviews = reviews.iloc[indices]
# Check your answer
q5.check()
sample_reviews
cols = ['country', 'province', 'region_1', 'region_2']
indices = [0, 1, 10, 100]
df = reviews.loc[indices, cols]

# Check your answer
q6.check()
df
df = reviews.loc[:99, ['country','variety']]

#cols = ['country', 'variety']
#df = reviews.loc[:99, cols]
                #or
#cols_idx = [0, 11]
#df = reviews.iloc[:100, cols_idx]

# Check your answer
df
italian_wines = reviews[reviews.country == 'Italy']

#Check your answer

q8.check()
q8.hint()
top_oceania_wines = reviews.loc[(reviews.country.isin(['Australia', 'New Zealand'])) & (reviews.points >= 95)]

# Check your answer
q9.check()
top_oceania_wines
q9.hint()
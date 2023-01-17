import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.indexing_selecting_and_assigning import *
print("Setup complete.")
reviews.head(3)
# Your code here
desc = reviews['description']

q1.check()
# Uncomment the line below to see a solution
q1.solution()
first_description = reviews['description'][0]

q2.check()
first_description
#q2.solution()
first_row = reviews.iloc[0]

q3.check()
first_row
q3.solution()
first_descriptions = reviews.description.iloc[:10]

q4.check()
first_descriptions
#q4.solution()
sample_reviews = reviews.iloc[[1,2,3,5,8]]

q5.check()
sample_reviews
#q5.solution()
df = reviews[['country','province','region_1','region_2']].iloc[[0,1,10,100]]

q6.check()
df
#q6.solution()
df = reviews[['country','variety']].iloc[0:100]

q7.check()
df
#q7.solution()
italian_wines = reviews[reviews.country == 'Italy']
print(len(reviews.country))
print(len(italian_wines))

q8.check()
q8.solution()
top_oceania_wines = reviews[reviews.points >= 95 &
    (reviews.country.isin(['Australia','New Zealand']))]

q9.check()
top_oceania_wines
q9.solution()
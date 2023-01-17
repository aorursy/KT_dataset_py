import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.indexing_selecting_and_assigning import *
print("Setup complete.")
reviews.head()
# Your code here
desc = reviews.description

q1.check()
# Uncomment the line below to see a solution
#q1.solution()
first_description = desc.values[0]

q2.check()
first_description
#q2.solution()
first_row = reviews.iloc[0]

q3.check()
first_row
q3.solution()
#first_descriptions = pd.Series( reviews.description.values[:10])
first_descriptions =  reviews.description.iloc[:10]
q4.check()
first_descriptions
q4.solution()
indices = [1, 2, 3, 5, 8]
sample_reviews = reviews.iloc[indices]
#sample_reviews = reviews.iloc[0:6] + reviews.iloc[9]

q5.check()
sample_reviews
q5.solution()
ind =[0,1,10,100]
cl = ['country','province','region_1', 'region_2']
df = reviews.loc[ind,cl]

q6.check()
df
#q6.solution()
cl = ['country', 'variety']
df = reviews.loc[0:99,cl]

q7.check()
df
q7.solution()
italian_wines =reviews[reviews.country == 'Italy']

q8.check()
italian_wines
q8.solution()
top_oceania_wines = reviews[reviews.country.isin(['Australia' , 'New Zealand']) & (reviews.points >= 95)]

q9.check()
top_oceania_wines

q9.solution()
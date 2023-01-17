import pandas as pd



reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

pd.set_option("display.max_rows", 5)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.indexing_selecting_and_assigning import *

print("Setup complete.")
reviews.head()
# Your code here

desc = reviews['description']



# Check your answer

q1.check()
#q1.hint()

#q1.solution()
first_description = reviews['description'][0]



# Check your answer

q2.check()

first_description
#q2.hint()

#q2.solution()
first_row = reviews.loc[0]



# Check your answer

q3.check()

first_row
#q3.hint()

#q3.solution()
first_descriptions = reviews['description'][0:10]



# Check your answer

q4.check()

first_descriptions
#q4.hint()

#q4.solution()
sample_reviews = reviews.loc[[1,2,3,5,8]]



# Check your answer

q5.check()

sample_reviews
#q5.hint()

#q5.solution()
data = {'country':['Italy','Portugal','US','US'],

       'province':['Sicily & Sardinia','Douro','California','New York'],

       'region_1':pd.Series(['Etna','Napa Valley','Finger Lakes'],index=[0,10,100]),

       'region_2':pd.Series(['Napa','Finger Lakes'],index=[10,100])}

df = pd.DataFrame(data,index=[0,1,10,100])



# Check your answer

q6.check()

df
#q6.hint()

#q6.solution()
df = reviews[['country','variety']][0:100]



# Check your answer

q7.check()

df
#q7.hint()

#q7.solution()
italian_wines = reviews[reviews.country == 'Italy']



# Check your answer

q8.check()
#q8.hint()

#q8.solution()
top_oceania_wines = reviews[reviews['country'].isin(['Australia','New Zealand'])][reviews['points'] >=95]



# Check your answer

q9.check()

top_oceania_wines
#q9.hint()

# q9.solution()
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

#q1.hint()

#q1.solution()
first_description = desc[0]



#alternatives

#reviews.description[0]

#reviews["description"].iloc[0]

#reviews["description"].loc[0]



# Check your answer

q2.check()

first_description
#q2.hint()

#q2.solution()
first_row = reviews.iloc[0]



#Alternatives

#reviews.loc[0]



# Check your answer

q3.check()

first_row
#q3.hint()

#q3.solution()
first_descriptions = reviews.loc[ : 9 , "description"] # text based



#Alternatives

#desc.iloc[:10]

#reviews.description.iloc[:10] - position based

#desc.head(10)



# Check your answer

q4.check()

first_descriptions
#q4.hint()

#q4.solution()
sample_reviews = reviews.iloc[[1,2,3,5,8]]



# Check your answer

q5.check()

sample_reviews
#q5.hint()

#q5.solution()
df = reviews.loc[[0,1,10,100],["country","province","region_1","region_2"]]



# Check your answer

q6.check()

df
#q6.hint()

#q6.solution()
df = reviews.loc[:99,["country","variety"]]



#Alternatives





# Check your answer

q7.check()

df
#q7.hint()

#q7.solution()
italian_wines = reviews.loc[reviews.country == "Italy"]



# Check your answer

q8.check()
#q8.hint()

#q8.solution()
top_oceania_wines = reviews.loc[(reviews.points >=95) & (reviews.country.isin(["Australia","New Zealand"]))]



# Check your answer

q9.check()

top_oceania_wines
#q9.hint()

#q9.solution()
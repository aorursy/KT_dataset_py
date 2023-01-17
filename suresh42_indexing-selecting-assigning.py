import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.indexing_selecting_and_assigning import *
print("Setup complete.")
reviews.head()
# Your code here
desc = reviews['description']
desc
#q1.hint()
#q1.solution()
first_description = reviews['description'][0]

first_description
#q2.hint()
#q2.solution()
first_row = reviews[0]
first_row
#q3.hint()
#q3.solution()
first_descriptions = reviews['description'][0:10:1]
first_descriptions
#q4.hint()
#q4.solution()
sample_reviews = reviews[0:9:1].drop([0,4,6,7])
sample_reviews
#q5.hint()
#q5.solution()

df = pd.DataFrame({'Country':['Italy','Portugal','US','US'],'Province':['Sicily','Dourto','california','Newyork'],'region_1':['Etna',0,'NapaValley','FingerLakes'],'region_2':['0','0','Napa','Fingerlakes']},columns=['Country','Province','region_1','region_2'],index=(0,1,10,100))
df
#q6.hint()
#q6.solution()
df = ____

q7.check()
df
#q7.hint()
#q7.solution()
italian_wines = ____

q8.check()
#q8.hint()
#q8.solution()
top_oceania_wines = ____

q9.check()
top_oceania_wines
#q9.hint()
#q9.solution()
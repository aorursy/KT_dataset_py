import pandas as pd
pd.set_option("display.max_rows", 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.summary_functions_and_maps import *
print("Setup complete.")

reviews.head()
median_points = reviews.points.median()

q1.check()
# Uncomment the line below to see a solution
#q1.solution()
countries = reviews.country.unique()

q2.check()
#q2.solution()


reviews_per_country = reviews.country.value_counts()

q3.check()
q3.solution()
centered_price = reviews.price - reviews.price.mean()

q4.check()
#q4.solution()
val = reviews.points / reviews.price
#val.max()
bargain_wine = reviews.title.iloc[val.idxmax()]

q5.check()
#q5.solution()
#print(reviews.description.values.size)
word1 = 'fruity'
word1count = 0
word2 = 'tropical'
word2count = 0
for i,des in enumerate(reviews.description.values):
    if word1 in des: word1count +=1
    if word2 in des: word2count +=1
#n_trop = reviews.description.map(lambda desc: "tropical" in desc).sum()
#n_fruity = reviews.description.map(lambda desc: "fruity" in desc).sum()        
#print (word2count)
descriptor_counts = pd.Series([word2count,word1count], index =[word2,word1])

q6.check()
q6.solution()
def star_true(row):
    if (row.country == 'Canada'): 
        return 3
    elif row.points >= 95: return 3
    elif (row.points < 95) and (row.points >= 85): return 2
    elif row.points < 85: return 1
    

n_trop = reviews.apply(star_true, axis= 'columns')
#n_trop = reviews.country.apply(star_bought)
star_ratings = n_trop
print (reviews.country.values)
q7.check()
q7.solution()
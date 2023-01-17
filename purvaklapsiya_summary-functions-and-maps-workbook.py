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
print("US",reviews_per_country['US'])
print(reviews_per_country)
q3.check()
#q3.solution()
mn = reviews.price.mean()
centered_price = reviews.price.map(lambda x: x - mn)

q4.check()
#q4.solution()
reviews['ratio'] = pd.Series(reviews.points/reviews.price)
review_temp = reviews[['ratio']]
idx = review_temp.idxmax()

bargain_idx = (reviews.points / reviews.price).idxmax()
print(bargain_idx)
bargain_wine = reviews.loc[int(idx), 'title']
# reviews.head()
q5.check()
q5.solution()
trop_count = reviews.description.map(lambda x: "tropical" in x).sum()
fruit_count = reviews.description.map(lambda x: "fruity" in x).sum()
descriptor_counts = pd.Series([trop_count,fruit_count], index=['tropical','fruity']) 

q6.check()
q6.solution()
def check_c(row):
    if row.points >= 95:
        row.points = 3
        return row
    elif  85<=row.points and row.points < 95:
        row.points = 2
        return row
    else:
        row.points = 1
        return row
x = reviews.apply(check_c, axis = 'columns')
star_rating = x['points']
print(reviews.head(10))
print(x.head(10))
# star_rating
q7.solution()



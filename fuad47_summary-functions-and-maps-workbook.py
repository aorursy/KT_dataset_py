import pandas as pd

pd.set_option("display.max_rows", 5)

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.summary_functions_and_maps import *

print("Setup complete.")



reviews.head()
median_points = reviews.points.median()



q1.check()
#q1.hint()

#q1.solution()
countries = reviews.country.unique()



q2.check()
q2.hint()

#q2.solution()
#reviews_per_country = reviews.country.count() map(country, lambda: x.count())

reviews_per_country = reviews.country.value_counts()

q3.check()
q3.hint()

#q3.solution()
centered_price = reviews.price-reviews.price.mean()



q4.check()
#q4.hint()

#q4.solution()
bargain_max = (reviews.points/reviews.price).idxmax()

#bargain_wine = (reviews.points/reviews.price).idxmax()

bargain_wine= reviews.loc[bargain_max,'title']



q5.check()
q5.hint()

q5.solution()
#descriptor_counts = pd.Series(reviews.description.sum())

n_trop = reviews.description.map(lambda desc: "tropical" in desc).sum()

n_fruity = reviews.description.map(lambda desc: "fruity" in desc).sum()

descriptor_counts = pd.Series([n_trop, n_fruity], index=['tropical', 'fruity'])





q6.check()
q6.hint()

q6.solution()
def ratings(row):

    if row.points<85:

        return 1

    elif row.points<=95:

        return 2

    else: 

        return 3

star_ratings =reviews.apply(ratings,axis='columns') 

type(star_ratings)







q7.check()
q7.hint()

q7.solution()
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
#q3.solution()
reviews_price_mean = reviews.price.mean()
centered_price = reviews.price.map(lambda x : x - reviews_price_mean)

q4.check()
#q4.solution()
bargain_idx = (reviews.points / reviews.price).idxmax()
bargain_wine = reviews.loc[bargain_idx,"title"]

q5.check()
q5.solution()
n_trop = reviews.description.map(lambda desc: "tropical" in desc).sum()
n_fruit = reviews.description.map(lambda desc: "fruity" in desc).sum()
descriptor_counts = pd.Series([n_trop,n_fruit], index = ["tropical","fruity"])

q6.check()
q6.solution()
def rating(row):
    if row.points >= 95:
        return 3
    elif 85 <= row.points < 95:
        return 2
    else:
        if row.country == 'Canada':
            return 3
        else:
            return 1

stars =  reviews.apply(rating, axis = "columns")
star_ratings = pd.Series(stars)

q7.check()
#q7.solution()
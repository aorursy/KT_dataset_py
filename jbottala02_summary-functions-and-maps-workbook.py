import pandas as pd
pd.set_option("display.max_rows", 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.summary_functions_and_maps import *
print("Setup complete.")

reviews.head()
median_points = reviews.points.median()

q1.check()
median_points
# Uncomment the line below to see a solution
#q1.solution()
countries = reviews.country.unique()

q2.check()
countries
#q2.solution()
reviews_per_country = reviews.country.value_counts()

q3.check()
reviews_per_country
#q3.solution()
review_price_mean = reviews.price.mean()
centered_price = reviews.price.map(lambda p: p - review_price_mean)

q4.check()
centered_price
#q4.solution()
idx = (reviews.points / reviews.price).idxmax()
bargain_wine = reviews.loc[idx, 'title']

q5.check()
bargain_wine
#q5.solution()
tropical = reviews.description.map(lambda des: "tropical" in des).sum()
fruity = reviews.description.map(lambda des1: "fruity" in des1).sum()
descriptor_counts = pd.Series([tropical, fruity], index=['tropical', 'fruity'])

q6.check()
descriptor_counts
#q6.solution()
def star(row):
    if row.country == 'Canada':
        return 3
    elif row.points >= 95:
        return 3
    elif row.points >= 85:
        return 2
    else:
        return 1

star_ratings = reviews.apply(star, axis='columns')

q7.check()
#q7.solution()
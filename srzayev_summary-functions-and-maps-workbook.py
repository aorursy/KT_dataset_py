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
countries = set(reviews.country)

q2.check()
#q2.solution()
reviews_per_country = reviews.country.value_counts()

q3.check()
#q3.solution()
centered_price = reviews.price.mean()
centered_price = reviews.price.map(lambda x: x - centered_price)

q4.check()
#q4.solution()
bargain_wine = (reviews.points / reviews.price).idxmax()
bargain_wine = reviews.loc[bargain_wine, 'title']

q5.check()
#q5.solution()
tropical = reviews.description.map(lambda x: "tropical" in x).sum()
fruity = reviews.description.map(lambda x: "fruity" in x).sum()
descriptor_counts = pd.Series([tropical, fruity], index=['tropical', 'fruity'])
print(descriptor_counts)
q6.check()
#q6.solution()
def get_stars(row):
    if row.country == 'Canada':
        return 3
    elif row.points >= 95:
        return 3
    elif row.points >= 85:
        return 2
    else:
        return 1
star_ratings = reviews.apply(get_stars, axis="columns")

q7.check()
#q7.solution()
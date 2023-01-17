import pandas as pd

pd.set_option("display.max_rows", 5)

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.summary_functions_and_maps import *

print("Setup complete.")



reviews.head()
median_points = reviews.points.median()

median_points
countries = reviews.country.unique()

countries
reviews_per_country = reviews.country.value_counts()

reviews_per_country
centered_price = reviews.price - reviews.price.mean()

centered_price
bargain_idx = (reviews.points / reviews.price).idxmax()

bargain_wine = reviews.loc[bargain_idx, 'title']

bargain_wine
n_trop = reviews.description.map(lambda desc: "tropical" in desc).sum()

n_fruity = reviews.description.map(lambda desc: "fruity" in desc).sum()

descriptor_counts = pd.Series([n_trop, n_fruity], index=['tropical', 'fruity'])

descriptor_counts
def stars(row):

    if row.country == 'Canada':

        return 3

    elif row.points >= 95:

        return 3

    elif row.points >= 85:

        return 2

    else:

        return 1



star_ratings = reviews.apply(stars, axis='columns')

star_ratings
import pandas as pd

pd.set_option("display.max_rows", 5)

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.summary_functions_and_maps import *

print("Setup complete.")



reviews.head()
median_points = reviews['points'].median(axis=0)



q1.check()
#q1.hint()

# q1.solution()
countries = reviews.country.unique()

print(countries)

q2.check()
#q2.hint()

# q2.solution()
reviews_per_country = reviews.country.value_counts()



q3.check()
#q3.hint()

# q3.solution()
centered_price = reviews['price'] - reviews['price'].mean()



q4.check()
#q4.hint()

# q4.solution()
bargain_idx = (reviews['points']/reviews['price']).idxmax()

bargain_wine= reviews.loc[bargain_idx,'title']

bargain_wine

q5.check()
# q5.hint()

# q5.solution()
tropical_count = reviews.description.map(lambda desc: 'tropical' in desc).sum()

fruity_count = reviews.description.map(lambda desc: 'fruity' in desc).sum()



descriptor_counts = pd.Series([tropical_count,fruity_count],index=['tropical','fruity'])

print(descriptor_counts)



q6.check()
# q6.hint()

# q6.solution()
def rate_star(x):

    if x.points >= 95  or x.country == 'Canada':

        return 3

    elif x.points >= 85 and x.points < 95:

        return 2

    else:

        return 1

    

star_ratings = reviews.apply(rate_star,axis=1)



print(star_ratings.head())

q7.check()
# q7.hint()

# q7.solution()



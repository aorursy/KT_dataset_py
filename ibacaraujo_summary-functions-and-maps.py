import pandas as pd

pd.set_option("display.max_rows", 5)

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.summary_functions_and_maps import *

print("Setup complete.")



reviews.head()
median_points = reviews.points.median()

print(median_points)



q1.check()
#q1.hint()

#q1.solution()
countries = reviews.country.unique()

print(countries)



q2.check()
#q2.hint()

#q2.solution()
reviews_per_country = reviews.country.value_counts()

print(reviews_per_country)



q3.check()
#q3.hint()

#q3.solution()
centered_price = reviews.price - reviews.price.mean()

print(centered_price)



q4.check()
#q4.hint()

#q4.solution()
bargain_idx = (reviews.points / reviews.price).idxmax()

bargain_wine = reviews.loc[bargain_idx, 'title']



q5.check()
q5.hint()

q5.solution()
check_tropical = reviews.description.map(lambda d : "tropical" in d)

check_fruity = reviews.description.map(lambda d : "fruity" in d)

n_tropical = check_tropical.sum()

n_fruity = check_fruity.sum()

descriptor_counts = pd.Series([n_tropical, n_fruity], index=['tropical', 'fruity'])

print(descriptor_counts)



q6.check()
q6.hint()

q6.solution()
def translate_to_star_ratings(row):

    if row.country == 'Canada':

        return 3

    if row.points > 95:

        return 3

    elif row.points >= 85 and row.points < 95:

        return 2

    else:

        return 1



star_ratings = reviews.apply(translate_to_star_ratings, axis='columns')



q7.check()
#q7.hint()

#q7.solution()
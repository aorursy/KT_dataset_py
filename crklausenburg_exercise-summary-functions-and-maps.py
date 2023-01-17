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
#q2.hint()

#q2.solution()
reviews_per_country = reviews.country.value_counts()



q3.check()
#q3.hint()

#q3.solution()
centered_price = reviews.price - reviews.price.mean()



q4.check()
#q4.hint()

#q4.solution()
bargain_wine = reviews.title.iloc[(reviews.points / reviews.price).idxmax()]

#print(bargain_wine)

q5.check()
q5.hint()

#q5.solution()
tropical = reviews.description.str.contains('tropical').value_counts().loc[True]

fruity = reviews.description.str.contains('fruity').value_counts().loc[True]

descriptor_counts = pd.Series([tropical, fruity], index=['tropical', 'fruity'])

#print(descriptor_counts)



""""

n_trop = reviews.description.map(lambda desc: "tropical" in desc).sum()

n_fruity = reviews.description.map(lambda desc: "fruity" in desc).sum()

descriptor_counts2 = pd.Series([n_trop, n_fruity], index=['tropical', 'fruity'])

print(descriptor_counts2)

"""

q6.check()
q6.hint()

#q6.solution()
def stars(row):

    stars = 0

    if row.country == 'Canada':

        return 3

    if row.points >= 95:

        return 3

    elif (row.points < 95 and row.points >= 85):

        return 2

    else:

        return 1



star_ratings = reviews.apply(stars, axis='columns')



q7.check()
#q7.hint()

#q7.solution()
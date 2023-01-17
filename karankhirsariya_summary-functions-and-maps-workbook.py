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
reviews_price_mean = reviews.price.mean()

centered_price = reviews.price.map(lambda p : p - reviews_price_mean)



q4.check()
#q4.hint()

#q4.solution()
points_to_price = reviews.points / reviews.price

bargain_wine = reviews.title[points_to_price.idxmax()]



q5.check()
#q5.hint()

#q5.solution()
nTropical = reviews.description.map(lambda desc : 'tropical' in desc).sum()

nFruity = reviews.description.map(lambda desc: 'fruity' in desc).sum()

descriptor_counts = pd.Series([nTropical,nFruity],index=['tropical','fruity'])



q6.check()
#q6.hint()

#q6.solution()
def rate(point):

    if point >= 95:

        return 3

    elif point >= 85:

        return 2

    else:

        return 1

star_ratings = reviews.points.map(rate)



q7.check()
#q7.hint()

#q7.solution()
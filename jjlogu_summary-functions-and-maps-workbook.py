import pandas as pd

pd.set_option("display.max_rows", 15)

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.summary_functions_and_maps import *

print("Setup complete.")



reviews.head()
reviews.describe()
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



reviews_per_country
#q3.hint()

#q3.solution()
mean_price = reviews.price.mean()

centered_price = reviews.price.map(lambda p: p-mean_price)



q4.check()



centered_price
#q4.hint()

#q4.solution()
(reviews.points/reviews.price).idxmax()
bargain_wine = reviews.loc[(reviews.points/reviews.price).idxmax(), 'title']



q5.check()



bargain_wine
#q5.hint()

q5.solution()
tropical= reviews.description.map(lambda d: "tropical" in d)

fruity =  reviews.description.map(lambda d: "fruity" in d)



print(type(tropical))

print(type(fruity))

print("asdfa asdf".count("asdf"))

print("asdf" in "asdfa asdf")

descriptor_counts = pd.Series([tropical.sum(),fruity.sum()], index=["tropical", 'fruity'])



q6.check()

descriptor_counts
q6.hint()

q6.solution()
reviews.head()
reviews['star']="0"
def calculate_star(row):

    if "Canada" == row.country or row.points >= 95:

        row.star = 3

    elif row.points < 95 and row.points >=85:

        row.star = 2

    else:

        row.star = 1

    return row
r = reviews.apply(calculate_star, axis='columns')
r[reviews.points >= 95]


star_ratings = r.star



q7.check()
q7.hint()

q7.solution()
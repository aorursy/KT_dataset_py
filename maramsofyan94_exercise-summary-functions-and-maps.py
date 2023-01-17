import pandas as pd

pd.set_option("display.max_rows", 5)

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.summary_functions_and_maps import *

print("Setup complete.")



reviews.head()
median_points = reviews.points.median()

#median_points = reviews['points'].median()





q1.check()

median_points
#q1.hint()

#q1.solution()
countries = reviews.country.unique()



q2.check()

countries
#q2.hint()

#q2.solution()
reviews_per_country = reviews.country.value_counts()



q3.check()

reviews_per_country
#q3.hint()

#q3.solution()
centered_price = reviews.price - reviews.price.mean()



q4.check()

centered_price
#q4.hint()

#q4.solution()


bargain_idx = (reviews.points / reviews.price).idxmax()

bargain_wine = reviews.loc[bargain_idx, 'title']



q5.check()

bargain_wine
n1 = reviews.price

n2 = reviews.points
(n1/n2).idxmax()
print(bargain_idx)

bargain_idx1 = (reviews.points).idxmax()

bargain_idx2 = (reviews.price).idxmax()

print(bargain_idx1)

print(bargain_idx2)
#q5.hint()

#q5.solution()
reviews.description[0]
'fruit' in reviews.description[0]
'maram' in reviews.description[0]
sum([True, False, True, True])
n_trop = reviews.description.map(lambda desc: "tropical" in desc).sum()

n_fruity = reviews.description.map(lambda desc: "fruity" in desc).sum()

descriptor_counts = pd.Series([n_trop, n_fruity], index=['tropical', 'fruity'])



q6.check()

descriptor_counts
#q6.hint()

#q6.solution()


def funs(row):

    if row.country == 'Canada':

        return 3

    elif row.points >= 95:

        return 3

    elif row.points >= 85:

        return 2

    else:

        return 1



star_ratings = reviews.apply(funs, axis='columns')



q7.check()

star_ratings
#q7.hint()

#q7.solution()
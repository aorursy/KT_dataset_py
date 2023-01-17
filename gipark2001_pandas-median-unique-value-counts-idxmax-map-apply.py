import pandas as pd

pd.set_option("display.max_rows", 5)

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

#a = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv")



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.summary_functions_and_maps import *

print("Setup complete.")



reviews.head()
median_points = reviews['points'].median() #____



q1.check()

median_points
#q1.hint()

#q1.solution()
countries = reviews['country'].unique() # ____



q2.check()

countries
#q2.hint()

# q2.solution()
reviews_per_country = reviews_per_country = reviews['country'].value_counts() # ____

print(reviews_per_country)

q3.check()

#q3.hint()

#q3.solution()
type(reviews.price), reviews.price.mean()
centered_price = reviews.price - reviews.price.mean() #____

print(centered_price)

q4.check()
#q4.hint()

#q4.solution()

bargain_idx = (reviews.points / reviews.price).idxmax()

print(bargain_idx)

bargain_wine = reviews.loc[bargain_idx, 'title']



#bargain_wine = ____



q5.check()

bargain_wine

bargain_idx
#type(reviews), reviews.columns, reviews.shape, 

type(reviews.iloc[bargain_idx]), reviews.iloc[bargain_idx].shape, reviews.iloc[bargain_idx]
reviews.head()
type(bargain_wine), type(bargain_idx), bargain_wine, bargain_idx
# q5.hint()

# q5.solution()
#descriptor_counts = ____

n_trop = reviews.description.map(lambda desc: "tropical" in desc).sum()

n_fruity = reviews.description.map(lambda desc: "fruity" in desc).sum()

descriptor_counts = pd.Series([n_trop, n_fruity], index=['tropical', 'fruity'])

q6.check()

descriptor_counts
# q6.hint()

# q6.solution()
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



q7.check()

star_ratings
# q7.hint()

# q7.solution()
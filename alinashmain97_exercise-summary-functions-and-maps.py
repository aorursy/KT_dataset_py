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
centered_price = reviews['price'] - reviews['price'].mean()



q4.check()
# q4.hint()

# q4.solution()
bargain_idx = (reviews.points / reviews.price).idxmax()

bargain_wine = reviews.loc[bargain_idx, 'title']

bargain_wine
reviews_copy = reviews.copy()

reviews_copy['points_to_price'] = reviews.points / reviews.price

max_val = reviews_copy['points_to_price'].max()

bargain_wine = reviews_copy.loc[reviews_copy['points_to_price'] == max_val].iloc[0].title



q5.check()
# q5.hint()

# q5.solution()
count_tropical = reviews['description'].str.contains('tropical').value_counts()[True]

count_fruity = reviews['description'].str.contains('fruity').value_counts()[True]

descriptor_counts = pd.Series([count_tropical, count_fruity], index=['tropical','fruity'])



q6.check()
#q6.hint()

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

star_ratings = reviews.apply(stars,axis='columns')



q7.check()
#q7.hint()

#q7.solution()
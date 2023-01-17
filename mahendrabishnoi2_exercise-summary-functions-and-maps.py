import pandas as pd

import numpy as np

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



q5.check()
# np.argmax(reviews.points / reviews.price)

# pd.Series.idxmax(reviews.points / reviews.price)
#q5.hint()

#q5.solution()
n_trop = reviews.description.map(lambda desc: "tropical" in desc).sum()

n_fruity = reviews.description.map(lambda desc: "fruity" in desc).sum()



descriptor_counts = pd.Series([n_trop, n_fruity], index=["tropical", "fruity"])



q6.check()
#q6.hint()

#q6.solution()
def create_ratings(row):

    if row.country == "Canada" or row.points >= 95:

        return 3

    elif row.points < 85:

        return 1

    else:

        return 2





star_ratings = reviews.apply(create_ratings, 'columns')



q7.check()
new_df
#q7.hint()

#q7.solution()
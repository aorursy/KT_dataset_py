import pandas as pd

pd.set_option("display.max_rows", 5)

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.summary_functions_and_maps import *

print("Setup complete.")



reviews.head()
median_points = reviews['points'].median()



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
bargain_wine = reviews['title'][(reviews.points/reviews.price).idxmax()]



q5.check()
#q5.hint()

#q5.solution()
descriptors = ['tropical', 'fruity']

tropical = sum(map(lambda x : 1 if descriptors[0] in x else 0, reviews['description']))

fruity = sum(map(lambda x : 1 if descriptors[1] in x else 0, reviews['description']))

descriptor_counts = pd.Series([tropical, fruity],index=[descriptors[0],descriptors[1]])



q6.check()
#q6.hint()

#q6.solution()
def wine_ratings(row):

    if row.points >= 95:

        stars = 3

    elif row.points >= 85 and row.points < 95:

        stars = 2

    else:

        stars = 1

        

    if row.country == 'Canada':

        stars = 3

    

    return stars



star_ratings = reviews.apply(wine_ratings,axis=1)



q7.check()
#q7.hint()

#q7.solution()
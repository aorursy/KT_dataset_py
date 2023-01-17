import pandas as pd

pd.set_option("display.max_rows", 5)

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.summary_functions_and_maps import *

print("Setup complete.")



reviews.head()
median_points = reviews.points.median()



q1.check()
q1.hint()

q1.solution()
countries = reviews.country.unique()



q2.check()
q2.hint()

q2.solution()
reviews_per_country = reviews.country.value_counts()



q3.check()
q3.hint()

q3.solution()
centered_price = reviews.price - reviews.price.mean()



q4.check()
q4.hint()

q4.solution()
bargain_wine_idx = (reviews.points/reviews.price).idxmax()

bargain_wine = reviews.loc[bargain_wine_idx, 'title']



q5.check()
q5.hint()

q5.solution()
reviews.head()
num_tropical = reviews.description.map(lambda d: 'tropical' in d).sum()

num_fruity = reviews.description.map(lambda d: 'fruity' in d).sum()



descriptor_counts = pd.Series(data = [num_tropical, num_fruity], index = ['tropical', 'fruity'])



q6.check()
q6.hint()

q6.solution()
reviews.head()
def stars(row):

    if row.points >= 95 or row.country == 'Canada':

        return 3

    elif 85 <= row.points < 95:

        return 2

    else: 

        return 1



star_ratings = reviews.apply(stars, axis = 'columns')



q7.check()

star_ratings.head()
q7.hint()

q7.solution()
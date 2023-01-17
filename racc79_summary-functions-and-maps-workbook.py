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
reviews_per_country = reviews['country'].value_counts()



q3.check()
#q3.hint()

#q3.solution()
centered_price = reviews.price - reviews.price.mean()



q4.check()
#q4.hint()

#q4.solution()
reviews['ratio'] = reviews['points']/reviews['price']

bargain_wine = reviews[reviews['ratio'] == reviews['ratio'].max()].iloc[0].title



q5.check()
#q5.hint()

#q5.solution()
trop = reviews.description.apply(lambda s: 'tropical' in s)

ctrop = len(trop[trop == True])



frut = reviews.description.apply(lambda s: 'fruity' in s)

cfrut = len(frut[frut == True])



descriptor_counts = pd.Series([ctrop,cfrut], index=['tropical', 'fruity'])

q6.check()
#q6.hint()

#q6.solution()
def f(c, p):

    s = 0

    if c == 'Canada': s = 3

    elif p >= 95: s = 3

    elif p >= 85: s = 2

    else: s = 1

    return s



star_ratings = reviews.apply(lambda r: f(r['country'], r['points']), axis=1)



q7.check()
#q7.hint()

#q7.solution()
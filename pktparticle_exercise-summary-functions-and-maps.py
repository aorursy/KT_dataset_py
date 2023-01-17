import pandas as pd

#pd.set_option("display.max_rows", 5)

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.summary_functions_and_maps import *

print("Setup complete.")



reviews.head(20)
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
mean_price=reviews.price.mean()

centered_price = reviews.price.map(lambda p: p-mean_price)

q4.check()
#q4.hint()

#q4.solution()
index= (reviews.points/reviews.price).idxmax()

bargain_wine=reviews.loc[index,'title']

q5.check()
#q5.hint()

#q5.solution()
import numpy as np

trop=reviews.description.map(lambda x: 'tropical' in x)

fruity=reviews.description.map(lambda x: 'fruity' in x)

trop=np.sum(trop)

fruity=np.sum(fruity)

descriptor_counts=pd.Series([trop,fruity], index=['tropical','fruity'])

q6.check()
#q6.hint()

#q6.solution()
def fun(row):

    if row.points>=95:

        row.points=3

    elif row.points>=85 and row.points<95:

        row.points=2

    elif row.country=='Canada':

        row.points=3

    else:

        row.points=1

    return row

val = reviews.apply(fun, axis='columns')

star_ratings=val.loc[:,'points']

q7.check()

#q7.hint()

#q7.solution()
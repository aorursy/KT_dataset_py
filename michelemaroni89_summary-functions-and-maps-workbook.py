import pandas as pd
pd.set_option("display.max_rows", 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.summary_functions_and_maps import *
print("Setup complete.")

reviews.head()
median_points = reviews.points.median()

q1.check()
# Uncomment the line below to see a solution
#q1.solution()
countries = reviews.country.unique()

q2.check()
#q2.solution()

reviews_per_country = reviews.country.value_counts()
q3.check()
reviews_per_country.head()

#q3.solution()

centered_price = reviews.price - reviews.price.mean()
q4.check()
#q4.solution()

bargain_wine = reviews.title.loc[(reviews.points / reviews.price).idxmax()]

q5.check()
#q5.solution()
trop = reviews.description.map(lambda x: 'tropical' in x)
fruity = reviews.description.map(lambda x : 'fruity' in x)
descriptor_counts = pd.Series([trop.sum(),fruity.sum()], index=['tropical','fruity'])

q6.check()
print(trop,fruity)
#q6.solution()
star_ratings = reviews.points.map(lambda x: 1 if x<85 else (2 if x<95 else(3)))


q7.check()
q7.solution()
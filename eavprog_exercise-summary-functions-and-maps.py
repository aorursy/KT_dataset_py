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

print(countries)



q2.check()
#q2.hint()

#q2.solution()
reviews_per_country = reviews.country.value_counts()



q3.check()



reviews_per_country
#q3.hint()

#q3.solution()
centered_price = reviews.price - reviews.price.mean()



q4.check()
#q4.hint()

#q4.solution()
bw = reviews.points / reviews.price



bargain_wine = reviews[bw == bw.max()]['title'].values[0]



bargain_wine



q5.check()
#q5.hint()

#q5.solution()
descriptor_counts = pd.Series( {wine : reviews.description.str.contains(wine).sum() for wine in 'tropical fruity'.split()})



print(descriptor_counts)



q6.check()
#q6.hint()

#q6.solution()
star_ratings = reviews.apply(lambda x: 3 if (x.points>=95 or x.country=='Canada') else 2 if x.points>=85 else 1,axis=1)



q7.check()
#q7.hint()

#q7.solution()
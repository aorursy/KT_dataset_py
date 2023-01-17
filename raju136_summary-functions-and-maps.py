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
centered_price = reviews.price- reviews.price.mean()



q4.check()
#q4.hint()

#q4.solution()
bargain_wine = reviews['title'][(reviews.points/reviews.price).idxmax()]

print(bargain_wine)

q5.check()
q5.hint()

q5.solution()
a=reviews.description.apply(lambda x: 'tropical' in x).sum()

b=reviews.description.apply(lambda x: 'fruity' in x).sum()

descriptor_counts = pd.Series([a,b],index=['tropical','fruity'])

print(descriptor_counts)

q6.check()
q6.hint()

q6.solution()
val=0

def get_star(row):

    if row.country=='Canada':

        return 3

    elif row.points>=95:

        return 3

    elif row.points>=85:

        return 2

    elif row.points<85:

        return 1

    

star_ratings =reviews.apply(get_star,axis=1)

print(type(star_ratings))



q7.check()
q7.hint()

q7.solution()
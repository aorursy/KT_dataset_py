import pandas as pd

pd.set_option("display.max_rows", 5)

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.summary_functions_and_maps import *

print("Setup complete.")



reviews.head()
median_points = reviews.points.median()



# Check your answer

q1.check()
#q1.hint()

#q1.solution()
countries = reviews.country.unique()



# Check your answer

q2.check()
#q2.hint()

#q2.solution()
reviews_per_country = pd.Series(reviews.country.value_counts())



# Check your answer

q3.check()
#q3.hint()

#q3.solution()
centered_price = reviews.price - reviews.price.mean()



# Check your answer

q4.check()
#q4.hint()

#q4.solution()
ratios = (reviews.points/reviews.price).idxmax()

bargain_wine = reviews.loc[ratios,'title']



# Check your answer

q5.check()
#q5.hint()

#q5.solution()
tropical = reviews.description.map(lambda x: 'tropical' in x)

fruity = reviews.description.map(lambda x:'fruity' in x)

descriptor_counts = pd.Series([tropical.sum(),fruity.sum()],index=['tropical','fruity'])



# Check your answer

q6.check()
#q6.hint()

#q6.solution()
def ratings(row):

    rat = 0

    if row.country=='Canada':

        rat = 3

    elif row.points>=95:

        rat = 3

    elif row.points>=85 and row.points<95:

        rat = 2

    else:

        rat = 1

    return rat

star_ratings = reviews.apply(ratings,axis='columns')



# Check your answer

q7.check()
#q7.hint()

#q7.solution()
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
countries = list(reviews.country.unique())



# Check your answer

q2.check()
#q2.hint()

#q2.solution()
reviews_per_country = reviews.country.value_counts()



# Check your answer

q3.check()
#q3.hint()

#q3.solution()
centered_price = reviews.price - reviews.price.mean()



# Check your answer

q4.check()
#q4.hint()

#q4.solution()
bargain_wine = reviews.iloc[(reviews.points / reviews.price).idxmax()].title



# Check your answer

q5.check()
# q5.hint()

# q5.solution()
descriptor_counts = pd.Series([reviews.description.map(lambda s: 'tropical' in s).sum(), 

                     reviews.description.map(lambda s: 'fruity' in s).sum()], 

                            index=['tropical', 'fruity'])



print(descriptor_counts)

# Check your answer

q6.check()
# q6.hint()

# q6.solution()
def calculate_star_rating(row):

    if row.country == 'Canada':

        return 3

    elif row.points >= 95:

        return 3

    elif row.points >= 85:

        return 2

    else:

        return 1

star_ratings = reviews.apply(calculate_star_rating, axis='columns')



# Check your answer

q7.check()
# q7.hint()

# q7.solution()
import pandas as pd

pd.set_option("display.max_rows", 5)

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.summary_functions_and_maps import *

print("Setup complete.")



reviews.head()
median_points =reviews.points.median()

print(median_points)



# Check your answer

q1.check()
#q1.hint()

#q1.solution()
countries = reviews.country.unique()

print(countries)





# Check your answer

q2.check()
#q2.hint()

#q2.solution()
reviews_per_country = reviews.country.value_counts()

print(reviews_per_country)





# Check your answer

q3.check()

#q3.hint()

#q3.solution()
centered_price = reviews.price.map(lambda p: p - reviews.price.mean())



# Check your answer

q4.check()
#q4.hint()

#q4.solution()
bargain_idx = (reviews.points / reviews.price).idxmax()

bargain_wine = reviews.loc[bargain_idx, 'title']



# Check your answer

q5.check()
#q5.hint()

#q5.solution()
trops = reviews.description.map(lambda description: 'tropical' in description).sum()

fruits = reviews.description.map(lambda description: 'fruity' in description).sum()

descriptor_counts = pd.Series([trops, fruits], index=['tropical','fruity'])



# Check your answer

q6.check()
#q6.hint()

#q6.solution()
def star_function(row):

    if row.country == 'Canada' or row.points >= 95:

        return 3

    elif row.points >= 85:

        return 2

    else:

        return 1



star_ratings = reviews.apply(star_function, axis='columns')



# Check your answer

q7.check()
#q7.hint()

#q7.solution()
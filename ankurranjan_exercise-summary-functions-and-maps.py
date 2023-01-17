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

# q3.solution()
centered_price = reviews.price.map(lambda p: p - reviews.price.mean())



# Check your answer

q4.check()
#q4.hint()

#q4.solution()
_row_lable_of_maximum = (reviews.points / reviews.price).idxmax()

print("Data Series  type \n", type(_row_lable_of_maximum ))

bargain_wine = reviews.loc[_row_lable_of_maximum, "title"]

print("Bargain Wine ", bargain_wine)

# Check your answer

q5.check()
# q5.hint()

# q5.solution()
tropical_count = reviews.description.map(lambda p: "tropical" in p).sum()

fruity_count = reviews.description.map(lambda p: "fruity" in p).sum()

descriptor_counts = pd.Series([tropical_count, fruity_count], index=['tropical', 'fruity'])

print(descriptor_counts)

# # Check your answer

q6.check()
# q6.hint()

# q6.solution()
def apply_rating(x, y):

    if y == "Canada":

        return 3

    elif x >= 95:

        return 3

    elif x >= 85:

        return 2

    else:

        return 1

    

star_ratings = reviews.apply(lambda p: apply_rating(p.points, p.country), axis="columns")



print(star_ratings)



# Check your answer

q7.check()
#q7.hint()

q7.solution()
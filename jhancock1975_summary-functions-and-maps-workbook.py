import pandas as pd
pd.set_option("display.max_rows", 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.summary_functions_and_maps import *
print("Setup complete.")

reviews.head()
reviews.describe(include='all')
median_points = reviews.describe().points['50%']

q1.check()
# Uncomment the line below to see a solution
#q1.solution()
countries = reviews.country.unique()

q2.check()
#q2.solution()
reviews_per_country = reviews.country.value_counts()

q3.check()
q3.solution()
centered_price = reviews.price - (reviews.describe().price['mean'])

q4.check()
#q4.solution()
points_to_price = reviews.points/reviews.price
bargain_wine = reviews.iloc[points_to_price.idxmax()]['title']

q5.check()
#q5.solution()
# it does not seem like this solution fits the requirements
# the requirements, to me, sound like they want the count
# of descriptions where the words tropical and fruity
# both occur in the description
n_trop = reviews.description.map(lambda desc: 'tropical' in desc).sum()
print(n_trop)
n_fruity = reviews.description.map(lambda desc: 'fruity' in desc).sum()
print(n_fruity)
descriptor_counts = pd.Series([n_trop,n_fruity], index=['tropical', 'fruity'])
print(descriptor_counts)
q6.check()
q6.solution()
def stars(row):
    if row.country == 'Canada':
        return 3
    elif row.points >= 95:
        return 3
    elif row.points >= 85:
        return 2
    else:
        return 1

star_ratings = reviews.apply(stars, axis='columns')

q7.check()
q7.solution()
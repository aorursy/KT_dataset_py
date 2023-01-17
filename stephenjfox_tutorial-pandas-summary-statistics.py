import pandas as pd

pd.set_option("display.max_rows", 5)

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.summary_functions_and_maps import *

print("Setup complete.")



reviews.head()
median_points = reviews['points'].median()



q1.check()
#q1.hint()

#q1.solution()
countries = reviews['country'].unique()



q2.check()
#q2.hint()

#q2.solution()
reviews_per_country = reviews['country'].value_counts()



q3.check()
#q3.hint()

#q3.solution()
centered_price = reviews['price'] - reviews['price'].mean()



q4.check()
#q4.hint()

#q4.solution()
reviews['points_to_price_ratio'] = reviews['points'] / reviews['price']

bargain_wine = reviews.iloc[reviews['points_to_price_ratio'].idxmax()]['title']

display(bargain_wine)

q5.check()
q5.hint()

q5.solution()
tropical_count = reviews['description'].str.contains('tropical').sum()

fruit_count = reviews['description'].str.contains('fruity').sum()

descriptor_counts = pd.Series([tropical_count, fruit_count], index=['tropical', 'fruity'])



q6.check()
# q6.hint()

# q6.solution()
def points_to_star(value: int) -> int:

    if value > 94:

        return 3

    elif 84 < value < 95:

        return 2

    else:

        return 1
star_ratings = reviews['points'].map(points_to_star)

star_ratings[reviews.country == 'Canada'] = 3



q7.check()
# q7.hint()

q7.solution()
import pandas as pd
pd.set_option("display.max_rows", 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.summary_functions_and_maps import *
print("Setup complete.")

reviews.head()
median_points = reviews['points'].median()

q1.check()
# Uncomment the line below to see a solution
#q1.solution()
countries = reviews['country'].unique()

q2.check()
#q2.solution()
reviews_per_country = reviews['country'].value_counts()

q3.check()
#q3.solution()
mean_value = reviews['price'].mean()
centered_price = reviews.price.map(lambda price: price-mean_value)

q4.check()
#q4.solution()
idx = (reviews['points'] / reviews ['price']).idxmax()
bargain_wine = reviews['title'].loc[idx]

q5.check()
#q5.solution()
def count_words(column,criteria):
    counter = 0
    for description in column:
        words = description.split()
        for word in words:
            if word.lower() == criteria:
                counter += 1
    return counter
tropical = reviews.description.map(lambda desc: 'tropical' in desc).sum()
fruity = reviews.description.map(lambda desc: 'fruity' in desc).sum()
descriptor_counts = pd.Series([tropical, fruity],index=['tropical','fruity'])
q6.check()
#q6.solution()
def convert_stars(row):
    stars = 0
    if 'Canada' == row.country:
        stars= 3
    elif row.points >= 95:
        stars= 3
    elif row.points >= 85:
        stars= 2
    else:
        stars= 1
    return stars

star_ratings = reviews.apply(convert_stars, axis='columns')

q7.check()
#q7.solution()
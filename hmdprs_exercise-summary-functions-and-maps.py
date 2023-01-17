from learntools.core import binder; binder.bind(globals())

from learntools.pandas.summary_functions_and_maps import *

print("Setup complete.")



import pandas as pd

pd.set_option("display.max_rows", 5)



reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
median_points = reviews['points'].median()



# check your answer

q1.check()
# q1.hint()

# q1.solution()
countries = reviews['country'].unique()



# check your answer

q2.check()
# q2.hint()

# q2.solution()
reviews_per_country = countries = reviews['country'].value_counts()



# check your answer

q3.check()
# q3.hint()

# q3.solution()
# 1st

# mean_price = reviews['price'].mean()  # NOTE: use this out of lambda

# centered_price = reviews['price'].map(lambda p: p - mean_price)



# 2nd

centered_price = reviews['price'] - reviews['price'].mean()



# check your answer

q4.check()
# q4.hint()

# q4.solution()
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.idxmax.html

max_value = (reviews['points'] / reviews['price']).idxmax()

bargain_wine = reviews.loc[max_value, 'title']



# check your answer

q5.check()
# q5.hint()

# q5.solution()
# reviews["description"].str.contains("tropical|fruity", na=False)

tropical = reviews['description'].map(lambda desc: 'tropical' in desc).sum()

fruity = reviews['description'].map(lambda desc: 'fruity' in desc).sum()

descriptor_counts = pd.Series([tropical, fruity], index=['tropical', 'fruity'])



# check your answer

q6.check()
# q6.hint()

# q6.solution()
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html

def stars(row):

    if row['country'] == 'Canada':

        return 3

    elif row['points'] >= 95:

        return 3

    elif row['points'] >= 85:

        return 2

    else:

        return 1



star_ratings = reviews.apply(stars, axis='columns')



# Check your answer

q7.check()
# q7.hint()

# q7.solution()
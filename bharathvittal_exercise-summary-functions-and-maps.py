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
q1.hint()

q1.solution()
countries = reviews.country.unique()



# Check your answer

q2.check()
q2.hint()

q2.solution()
# reviews.head()



reviews_per_country = reviews.country.value_counts()

# reviews_per_country = 



# Check your answer

q3.check()
q3.hint()

q3.solution()
# reviews.price.mean()



# centered_price = reviews.price.map(lambda x: x - reviews.price.mean())





centered_price =  reviews.price - reviews.price.mean()



# # Check your answer

q4.check()
q4.hint()

q4.solution()
# reviews.columns



bargain_idx = (reviews.points / reviews.price).idxmax()

# bargain_wine = reviews.loc[bargain_idx, 'title']

bargain_wine = reviews.loc[bargain_idx, 'title']





# Check your answer

q5.check()
q5.hint()

q5.solution()
reviews[reviews['description'].str.contains('fruity')].shape



reviews[reviews['description'].str.contains('tropical')].shape[0]



descriptor_counts = pd.Series({'tropical':reviews[reviews['description'].str.contains('tropical')].shape[0], 'fruity': reviews[reviews['description'].str.contains('fruity')].shape[0]})

descriptor_counts



# # Check your answer

q6.check()
q6.hint()

q6.solution()
reviews['star_ratings'] = 1



for i in range(len(reviews)):

    if (reviews['country'][i] == 'Canada') | reviews['points'][i] >= 95:

        reviews['star_ratings'][i] = 3

    elif reviews['points'][i] >= 85:

        reviews['star_ratings'][i] = 2

    else:

        reviews['star_ratings'][i] = 2

        

        

star_ratings = reviews['star_ratings']



# # Check your answer

q7.check()
q7.hint()

q7.solution()
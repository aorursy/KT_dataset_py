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
reviews_per_country = reviews.country.value_counts()



# Check your answer

q3.check()
#q3.hint()

q3.solution()
reviews.price.mean()

centered_price = reviews.price.apply(lambda x:x-reviews.price.mean())



# Check your answer

q4.check()
#q4.hint()

q4.solution()
high_p_p_r = (reviews.points / reviews.price).idxmax()

bargain_wine = reviews.loc[high_p_p_r, 'title']



# Check your answer

q5.check()
#q5.hint()

q5.solution()




n_tropical = reviews.description.apply(lambda x: 'tropical' in x).sum()

n_fruity = reviews.description.apply(lambda x: 'fruity' in x).sum()



descriptor_counts = pd.Series([n_tropical, n_fruity], index=['tropical','fruity'])



# Check your answer

q6.check()
#q6.hint()

q6.solution()
def give_stas(rows):

    if rows.points >= 95 or rows.country == 'Canada':

        return 3

    elif rows.points >=85:

        return 2

    else:

        return 1

    



star_ratings = reviews.apply(give_stas, axis='columns')



# Check your answer

q7.check()
#q7.hint()

q7.solution()
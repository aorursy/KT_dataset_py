import pandas as pd

pd.set_option("display.max_rows", 5)

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.summary_functions_and_maps import *

print("Setup complete.")



reviews.head()
median_points =reviews.points.median()



q1.check()
#q1.hint()

#q1.solution()
countries = reviews.country.unique()



q2.check()
#q2.hint()

#q2.solution()
reviews_per_country = reviews.country.value_counts()

reviews_per_country 

q3.check()
#q3.hint()

#q3.solution()
mean_price = reviews.price.mean()

centered_price = reviews.price- mean_price



q4.check()
#q4.hint()

#q4.solution()
ratio = (reviews.points/ reviews.price).idxmax()



bargain_wine = reviews.loc[ratio,'title']



q5.check()



bargain_wine
#q5.hint()

#q5.solution()
trop = reviews.description.apply(lambda desc: "tropical" in desc).sum()

fru = reviews.description.apply(lambda desc: "fruity" in desc).sum()

descriptor_counts = pd.Series([trop, fru], index=['tropical', 'fruity'])

descriptor_counts

q6.check()
#q6.hint()

#q6.solution()
#star_ratings = np.where(reviews.points >=95,"3 stars",(np.where(reviews.points >= 85 and reviews.points < 95,"2 stars","1 stars")))

def rating(row):

    if row.country == 'Canada':

        return 3

    elif row.points >= 95:

        return 3

    elif row.points >= 85:

        return 2

    else:

        return 1



star_ratings = reviews.apply(rating, axis='columns')

q7.check()
#q7.hint()

#q7.solution()
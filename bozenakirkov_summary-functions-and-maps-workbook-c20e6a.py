import pandas as pd

pd.set_option("display.max_rows", 5)

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.summary_functions_and_maps import *

print("Setup complete.")



reviews.head()
median_points = reviews.points.median()



q1.check()
#q1.hint()

#q1.solution()
countries = reviews.country.unique()



q2.check()
#q2.hint()

#q2.solution()
reviews_per_country = reviews.country.value_counts()



q3.check()
#q3.hint()

#q3.solution()
mean = reviews.price.mean()

centered_price = reviews.price.map(lambda p: p-mean)



q4.check()
#q4.hint()

#q4.solution()
id = (reviews.points/reviews.price).idxmax()

bargain_wine = reviews.title.loc[id]



q5.check()
#q5.hint()

#q5.solution()
num_fruity = reviews.description.map(lambda d: 'fruity' in d).sum()

num_tropical = reviews.description.map(lambda d: 'tropical' in d).sum()



descriptor_counts = pd.Series([num_tropical, num_fruity], index=['tropical', 'fruity'])



q6.check()
#q6.hint()

#q6.solution()
def star_rating(row): 

    star_rating_dict = {3:[95,96,97,98,99,100], 

                        2:[85,86,87,88,89,90,91,92,93,94], 

                        1:[80,81,82,83,84]}

    for key, value in star_rating_dict.items():

        if (row.points in value) and (row.country != 'Canada'):

            return key

        elif row.country == 'Canada':

            return 3

                                      

star_ratings = reviews.apply(star_rating, axis='columns')



q7.check()
#q7.hint()

#q7.solution()
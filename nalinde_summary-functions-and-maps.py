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
centered_price = reviews['price']- reviews['price'].mean()



q4.check()
#q4.hint()

#q4.solution()
rd=(reviews.points/reviews.price).idxmax()

bargain_wine = reviews.loc[rd,"title"]



q5.check()
#q5.hint()

#q5.solution()
d=reviews["description"]

di=d.map(lambda x:"tropical" in x).value_counts()

ds=d.map(lambda x:"fruity" in x).value_counts()

descriptor_counts = pd.Series([di[True],ds[True]],index=["tropical","fruity"])



q6.check()
#q6.hint()

#q6.solution()
def get_ratings(dat):

    star=0

    if dat["country"]=="Canada":

        star=3

    elif dat["points"]>=95:

        star=3

    elif dat["points"]>=85 and dat["points"]<95:

        star=2

    else:

        star=1

    return star    

star_ratings = reviews.apply(get_ratings,axis=1)



q7.check()
#q7.hint()

#q7.solution()
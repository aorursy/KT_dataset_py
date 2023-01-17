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
countries =reviews.country.unique()



q2.check()
#q2.hint()

#q2.solution()
reviews_per_country = reviews.country.value_counts()



q3.check()
#q3.hint()

#q3.solution()
centered_price = reviews.price - reviews.price.mean()

q4.check()
#q4.hint()

#q4.solution()
idxmax = (reviews.points/reviews.price).idxmax()

bargain_wine = reviews.loc[idxmax, "title"]

q5.check()

bargain_wine
#q5.hint()

#q5.solution()
tropical = reviews["description"].apply(lambda x:  x.count("tropical")).sum()

fruity = reviews["description"].apply(lambda x:  x.count("fruity")).sum()

descriptor_counts = pd.Series([tropical, fruity], index = ["tropical", "fruity"])

descriptor_counts

# q6.check()
tropical = reviews["description"].apply(lambda x:  "tropical" in x).sum()

fruity = reviews["description"].apply(lambda x:  "fruity" in x).sum()

descriptor_counts = pd.Series([tropical, fruity], index = ["tropical", "fruity"])

descriptor_counts

q6.check()
#q6.hint()

#q6.solution()
def define_star(x):

    star =1

    if x >= 95:

        star = 3

    elif x >= 85:

        star = 2

    return star

reviews["stars"] = reviews["points"].apply(lambda x: define_star(x))

reviews.loc[reviews["country"] == "Canada", "stars"] = 3

star_ratings = reviews.stars

q7.check()
#q7.hint()

q7.solution()
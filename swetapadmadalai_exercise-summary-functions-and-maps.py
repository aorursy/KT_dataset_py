import pandas as pd

pd.set_option("display.max_rows", 5)

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.summary_functions_and_maps import *

print("Setup complete.")



reviews.head()
import pandas as pd

reviews=pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

reviews

median_points = reviews.points.median()

median_points



# Check your answer

#q1.check()
#q1.hint()

#q1.solution()
countries = reviews.country.unique()



countries



# Check your answer

#q2.check()
#q2.hint()

#q2.solution()
reviews_per_country = reviews.country.value_counts()

reviews_per_country



# Check your answer

#q3.check()
#q3.hint()

#q3.solution()
mean_price=reviews.price.mean()

centered_price = reviews.price.map(lambda p: p-mean_price)



centered_price

# Check your answer

#q4.check()
#q4.hint()

#q4.solution()
bargain_wine = reviews.price.max()

bargain_wine

# Check your answer

#q5.check()
#q5.hint()

#q5.solution()
descriptor_counts = pd.Series(reviews.description.value_counts())

descriptor_counts

# Check your answer

#q6.check()
#q6.hint()

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

 

star_ratings

# Check your answer

#q7.check()
#q7.hint()

#q7.solution()
import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
reviews.head()
check_q1(pd.DataFrame())
reviews.head()
# Your code here
reviews.points.median()
# Your code here
reviews.country.unique()


# Your code here
reviews.country.value_counts()

# Your code here
median = reviews.price.median()
reviews.price.map(lambda x: x - median)
# Your code here
reviews.price.apply(lambda x: x - median)

reviews.loc[(reviews.points/reviews.price).idxmax()].title
# Your code here
tropical_frequency = sum(reviews.description.map(lambda x: 'tropical' in x))
fruity_frequency = sum(reviews.description.map(lambda x: 'fruity' in x))

fruity_vs_tropical = pd.Series([fruity_frequency, tropical_frequency], index=['fruity', 'tropical'], name='frequency')

fruity_vs_tropical
# Your code here
reviews_notnull = reviews[reviews.country.notnull() & reviews.variety.notnull()]

country_variety = reviews_notnull.apply(lambda x: x.country + ' - ' + x.variety, axis=1)

country_variety.value_counts()

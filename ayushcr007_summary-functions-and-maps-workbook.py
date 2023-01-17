import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
check_q1(reviews.points.median())
# Your code here
check_q2(reviews.country.unique())
# Your code here
check_q3(reviews.country.mode())
reviews.price
# Your code here
price_median = reviews.price.median()
check_q4(reviews.price.map(lambda x: x - price_median))
# Your code here
reviews[(reviews.points/reviews.price) == (reviews.points/reviews.price).max()]
#reviews.loc[(reviews.points/reviews.price).argmax()].title
#answer_q5()
# Your code here
#reviews.description
#reviews.description.map(lambda x: "tropical" in x).value_counts()
tropical_wine = reviews.description.map(lambda r: "tropical" in r).value_counts()
fruity_wine = reviews.description.map(lambda r: "fruity" in r).value_counts()
pd.Series([tropical_wine[True], fruity_wine[True]], index=['tropical', 'fruity'])
reviews.head()

# Your code here
import pandas as pd
df = pd.DataFrame(data = reviews, columns=['country','variety'])
df = df.dropna()
srs = df.country + '-' + df.variety
srs.value_counts()

#ans = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
#ans = ans.apply(lambda srs: srs.country + " - " + srs.variety, axis='columns')
#ans.value_counts()

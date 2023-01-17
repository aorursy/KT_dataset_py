import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
a1 = reviews.points.median()
check_q1(a1)

# Your code here
a2 = reviews.country.unique()
check_q2(a2)
# Your code here
a3 = reviews.country.value_counts()
check_q3(a3)
# Your code here
median = reviews.price.median()
a4 = reviews.price.map(lambda p: p - median)
check_q4(a4)
# indexing is off starting from exercise 5 - answer and chech for nr 5 work with index 6
#answer_q5() - this displays an answer to exercise 4

#my first solution
#points_to_price = pd.Series(reviews.points / reviews.price)
#points_to_price
#points_to_price.idxmax()
#reviews.iloc[points_to_price.idxmax()]
#a5 = reviews.iloc[points_to_price.idxmax()].title

#better solution, no need for extra pd.Series() call
a5 = reviews.loc[(reviews.points / reviews.price).idxmax()].title

check_q6(a5)



# my solution
#tropical = reviews.loc[reviews.description.str.contains("tropical")]
#fruity = reviews.loc[reviews.description.str.contains("fruity")]
#tropical_count = len(tropical)
#fruity_count = len(fruity)
#a6 = pd.Series([tropical_count, fruity_count], index=['tropical', 'fruity'])

#excersise solution
tropical_wine = reviews.description.map(lambda d: "tropical" in d).value_counts()
fruity_wine = reviews.description.map(lambda d: "fruity" in d).value_counts()
a6 = pd.Series([tropical_wine[True], fruity_wine[True]], index=["tropical", "fruity"])

check_q7(a6)


#answer_q7()

# my solution
notnan_reviews = reviews.loc[reviews.country.notnull() & reviews.variety.notnull()]
country_variety_combo = notnan_reviews.apply(lambda r: f"{r.country} - {r.variety}", axis=1)
a7 = country_variety_combo.value_counts()

check_q8(a7)
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
check_q3(reviews.country.value_counts())
# Your code here
check_q4(reviews.price.map(lambda x: x- reviews.price.median()))
# Your code here
def f(x) : return x- reviews.price.median()
check_q5(reviews.price.apply(f))
# Your code here
print(reviews.loc[(reviews.points/reviews.price).idxmax()].title)
check_q6(reviews.loc[(reviews.points/reviews.price).idxmax()].title)
# Your code here
tropical_wine = reviews.description.map(lambda x: "tropical" in x)
fruity_wine = reviews.description.map(lambda x: "fruity" in x)
check_q7(pd.Series([tropical_wine[tropical_wine==True].count(),fruity_wine[fruity_wine ==True].count()], index=['tropical','fruity']))
#tropical_wine = reviews.description.map(lambda r: "tropical" in r).value_counts()
#fruity_wine = reviews.description.map(lambda r: "fruity" in r).value_counts()
#fruity_wine.head()
#series=pd.Series([tropical_wine[True], fruity_wine[True]], index=['tropical', 'fruity'])
#print (series)
# Your code here
reviews2 = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
country_variety = reviews2.country+' - '+reviews2.variety
check_q8(country_variety.value_counts()) 
#answer_q8()
#ans = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
#ans = ans.apply(lambda srs: srs.country + " - " + srs.variety, axis='columns')
#ans.value_counts()
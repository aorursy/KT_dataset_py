import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
median_points = reviews.points.median()
check_q1(median_points)
check_q2(reviews.country.unique())
check_q3(reviews.country.value_counts())
price_median=reviews.price.median()
price_median
reviews.price.map(lambda p:p-price_median)
check_q4(reviews.price)
reviews.title[(reviews.points/reviews.price).idxmax()]


tropical = reviews.description.map(lambda p: "tropical" in p).value_counts()
fruity = reviews.description.map(lambda p: "fruity" in p).value_counts()
pd.Series([tropical[True], fruity[True]], index= ["tropical", "fruity"])




# function to count occurances of a specific word
#def count_tropical(p):
 #   if p.count("tropical") >0:
  #      return 1
   # else:
    #    return 0
#def count_fruity(q):
 #   if q.count("fruity") >0:
  #      return 1
   # else:
    #    return 0


#reviews["tropical_count"] = reviews.description.map(count_tropical)
#reviews["fruity_count"] = reviews.description.map(count_fruity)
#ss = pd.Series([reviews["tropical_count"].sum(),reviews["fruity_count"].sum()], index = ["tropical","fruity"])

#tropical_wine = reviews.description.map(lambda r: "tropical" in r).value_counts()
#fruity_wine = reviews.description.map(lambda r: "fruity" in r).value_counts()
#pd.Series([tropical_wine[True], fruity_wine[True]], index=['tropical', 'fruity'])

# generate a dataframe such that country and variety columns do not have null values.
country_variety = reviews[["country","variety"]].copy(deep=True)
type(country_variety)#.describe()
country_variety = country_variety.dropna()

#map(lambda country_variety.country)

answer_q8()




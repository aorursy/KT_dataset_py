import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
reviews.points.median()
reviews.country.unique()
reviews.country.value_counts()
price_med = reviews.price.median()
reviews.price.map(lambda m: m - price_med)
def minus_median(sries):
    sries.points = sries.points - price_med
    return sries

reviews.apply(minus_median, axis=1)
reviews.iloc[(reviews.points / reviews.price).idxmax()].title
fruit_punch = pd.Series(pd.DataFrame({"Tropical": reviews.description.map(lambda f: "tropical" in f), "Fruity": reviews.description.map(lambda t: "fruity" in t)}).apply(sum), index=["Tropical", "Fruity"])
fruit_punch
#trop_wine = reviews.description.map(lambda t: "tropical" in t).value_counts()
#fruit_wine = reviews.description.map(lambda f: "fruity" in f).value_counts()
#ans = pd.Series([trop_wine[True], fruit_wine[True]], index=["Tropical", "Fruity"])
non_null = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
country_types = pd.Series((non_null.country + " - " + non_null.variety).value_counts())
check_q7(country_types)
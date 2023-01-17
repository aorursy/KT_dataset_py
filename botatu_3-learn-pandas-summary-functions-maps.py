import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
answer_q8()
reviews.head()
reviews.points.median()
reviews.country.unique()
reviews.country.value_counts()
reviews_price_median = reviews.price.median()
reviews.price.map(lambda x: x - reviews_price_median)
reviews.loc[(reviews.points/reviews.price).idxmax()].title
tropical_wine = reviews.description.map(lambda x: 'tropical' in x).value_counts()
fruity_wine = reviews.description.map(lambda x: 'fruity' in x).value_counts()
pd.Series([tropical_wine[True], fruity_wine[True]], index=['Tropical', 'Fruity'])
country_to_variety = reviews[reviews.country.notnull() & reviews.variety.notnull()]
country_to_variety = country_to_variety.country + ' - ' + country_to_variety.variety
country_to_variety.value_counts()
import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
print (reviews['points'].median())
check_q1 (reviews['points'].median())
print (reviews['country'].unique())
check_q2 (reviews['country'].unique())
print (reviews['country'].value_counts() [:5])
check_q3 (reviews['country'].value_counts())
median_price = reviews['price'].median()
price_from_median = reviews['price'].map (lambda price: price - median_price)
print (price_from_median)
check_q4 (price_from_median)
points = reviews ['points']
prices = reviews.price
ratios = points / prices
ratios.nlargest (5)
# The check function only takes into account one of the two "best bargain" wines, so I don't use it here to check my answer, but print both the "best bargain" wines

best_bargains = reviews [ratios == ratios.max ()]
best_bargains = best_bargains.reset_index()
print (best_bargains.loc [0]['title'])
print (best_bargains.loc [1]['title'])
# From here on out, the `check` function doesn't work right anymore. This is exercise 5 but we use `check_q6` to check the answer. This pattern persists until the end of the workbook.
check_q6 (best_bargains.loc [0]['title'])
tropics = reviews['description'].map (lambda desc: 'tropical' in desc)
fruits = reviews.description.map (lambda desc: 'fruity' in desc)
count_tropics = tropics.value_counts ()
count_fruits = fruits.value_counts ()
count_fruitropics = pd.Series ([count_tropics [True], count_fruits [True]], index = ['tropical', 'fruity'])
check_q7 (count_fruitropics)
dropped_null = reviews.dropna (subset=['country', 'variety'])
countries = dropped_null ['country']
varieties = dropped_null ['variety']
country_varieties = countries + ' - ' + varieties
count_country_varieties = country_varieties.value_counts ()
check_q8 (count_country_varieties)
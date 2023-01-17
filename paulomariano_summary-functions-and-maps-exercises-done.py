import pandas as pd

pd.set_option('max_rows', 5)

import numpy as np

from learntools.advanced_pandas.summary_functions_maps import *



reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
reviews.points.median()
abc = reviews.country.unique()

x = pd.Series(abc, name = 'Páíses Repetidos')

x
reviews.country.value_counts()
# Aqui usamos para efeito de cálculo mas se fosse necessário mexer na dataset, precisamos usar uma função def. No tutorial desta aula tem.

remap_price = reviews.price.median()

reviews.price.map(lambda p: p - remap_price)
Ratio_Math = (reviews.points / reviews.price)

abc = Ratio_Math.sort_values(ascending = False)

index_tracked = abc.index



Counter = 0

for pts in abc:

    print(f'{pts} - {reviews.title[abc.index[Counter]]}')

    Counter += 1

    

    if Counter == 10:

        break



































# Your code here
# Your code here
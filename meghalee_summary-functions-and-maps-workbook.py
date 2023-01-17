import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
reviews['points'].median()
# Your code here
set(reviews['country'])
# Your code here
reviews['country'].mode()
# Your code here
median_price=reviews['price'].median()
median_price

reviews['price'].map(lambda v: v-median_price)
#print(answer_q4())
# Your code here
PointsRatio = (reviews.points/reviews.price)
PointsRatio.idxmax()
best_bargain=reviews.iloc[PointsRatio.idxmax()]
#reviews.info()
#PointsRatio = (reviews.points/reviews.price).idxmax()
#show = reviews.iloc[PointsRatio]
print(best_bargain['title'],'from',best_bargain['winery'], 'is the best bargain !!!')
# Your code here

#wine_tro_fr=reviews.description.map[lambda p:  True if "tropical" in p else False]
#wine_tro_fr


tropical = reviews.description.map(lambda p: True if "tropical" in p else False)
ts=tropical.value_counts()
#ts[True]

#type(ts)

fruity = reviews.description.map(lambda p: True if "fruity" in p else False)
fs=fruity.value_counts()
#fs
index=['tropical','fruity']

cs=pd.Series([ts[True],fs[True]],index)
cs
#type(fs)
# Your code here
#print(answer_q8())

#wine=reviews

#wine.head(10)

#print(answer_q8())
ans = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
srs=reviews
#type(srs)
srs_str= srs.country + " - " + srs.variety
srs_str
ans = ans.apply(lambda srs: srs.country + " - " + srs.variety, axis='columns')
ans.value_counts()
#reviews.info()
#ans=wine['country'].notnull() & wine['variety'].notnull()
#ans[ans.values==True].value_counts()
#index_true=ans[ans.values==True].index
#index_true.value_counts()
#rv_c=reviews.iloc[index_true]
#str_con=rv_c['country']+'-'+rv_c['variety']
#ans = ans.apply(lambda srs: str_con)
#ans
#wine_crt=ans.value_counts()
#wine_crt
 
#type(ans)
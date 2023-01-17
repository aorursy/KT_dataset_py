import pandas as pd
from learntools.advanced_pandas.grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
check_q1(pd.DataFrame())
# Your code here
reviews.head()
taster_groups=reviews.groupby('taster_twitter_handle').title.count()
taster_groups


# common_wine_reviewers = _______
# check_q1(common_wine_reviewers)
#print(answer_q1())
# Your code here
# best_wine = ______
# check_q2(best_wine)
#reviews.head()
best_wine=reviews.groupby('price').points.max().sort_index()
t=reviews.groupby('price').apply(lambda df: df.loc[df.points.idxmax()])
t
t.title[t['price']==240.0]

#print(answer_q2())
# Your code here
# wine_price_extremes = _____
# check_q3(wine_price_extremes)
reviews.groupby(['variety']).price.agg([min,max]).sort_values(by=['min','max'])
# Your code here
# reviewer_mean_ratings = _____
# check_q4(reviewer_mean_ratings)
avg_review=reviews.groupby(['taster_name']).points.mean()
avg_review
#type(_)
#print(answer_q4())
# Your code here
# wine_price_range = ____
# check_q5(wine_price_range)
#reviews.head()
reviews.groupby(['variety']).price.agg([min,max]).sort_values(by=['min','max'])

# Your code here
# country_variety_pairs = _____
# check_q6(country_variety_pairs)
reviews.groupby(['country', 'province']).title.count().sort_values(ascending=False)
#print(answer_q6())
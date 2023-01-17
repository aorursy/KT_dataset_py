import pandas as pd
from learntools.advanced_pandas.grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews
check_q1(pd.DataFrame())
# Your code here
common_wine_reviewers = reviews.taster_twitter_handle.value_counts()
check_q1(common_wine_reviewers)
#common_wine_reviewers.sum()
#len(reviews)
#reviews.taster_twitter_handle.isna().sum()==len(reviews)-common_wine_reviewers.sum()
#reviews.taster_name.isna().sum()
#common_wine_reviewers.index
#reviews.taster_twitter_handle.unique()
common_wine_reviewers
reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()
#This does not give the most common wine reviewers, they're unsorted
# Your code here
best_wine = reviews.groupby("price").points.max()
best_wine
check_q2(best_wine)
# Your code here
wine_price_extremes = pd.DataFrame({"min": reviews.groupby("variety").price.min(), "max": reviews.groupby("variety").price.max()})
check_q3(wine_price_extremes)
#wine_price_extremes
# Your code here
reviewer_mean_ratings = reviews.groupby("taster_name").points.mean()
check_q4(reviewer_mean_ratings)
# Your code here
wine_price_range = wine_price_extremes.sort_values(["min","max"], ascending=False)
check_q5(wine_price_range)
#wine_price_range
# Your code here
reviews['n'] = 0
#count the number of 0s in column n for each country/variety pairing this is the number of rows
#I did reviews.groupby(["country","variety"]).count().sort_values(by="n",ascending=False) and got a dataframe
#which is obviously wrong but it does give the count for each column
country_variety_pairs = reviews.groupby(["country","variety"]).n.count().sort_values(ascending=False)
check_q6(country_variety_pairs)
reviews.groupby(["country","variety"]).count().sort_values(by="n",ascending=False)
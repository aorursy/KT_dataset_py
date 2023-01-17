import pandas as pd
from learntools.advanced_pandas.grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
check_q1(pd.DataFrame())
common_wine_reviewers = reviews.groupby("taster_twitter_handle")["taster_twitter_handle"].count()
check_q1(common_wine_reviewers)
best_wine = reviews.groupby("price")["points"].max()
check_q2(best_wine)
wine_price_extremes = reviews.groupby("variety")["price"].agg([min, max])
check_q3(wine_price_extremes)
reviewer_mean_ratings = reviews.groupby("taster_name")["points"].mean()
check_q4(reviewer_mean_ratings)
wine_price_range = reviews.groupby("variety")["price"].agg([min, max]).sort_values(["min", "max"], ascending=False)
check_q5(wine_price_range)
country_variety_pairs = reviews.groupby(["country", "variety"])["title"].count().sort_values(ascending=False)
check_q6(country_variety_pairs)

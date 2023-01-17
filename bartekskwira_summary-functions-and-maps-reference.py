import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
reviews = pd.read_csv("../input/winemag-data-130k-v2.csv", index_col=0)
reviews.head()
reviews.points.describe()
reviews.taster_name.describe()
reviews.points.mean()
reviews.taster_name.unique()
reviews.taster_name.value_counts()
review_points_mean = reviews.points.mean()
reviews.points.map(lambda p: p - review_points_mean)
def remean_points(srs):
    srs.points = srs.points - review_points_mean
    return srs

reviews.apply(remean_points, axis='columns')
review_points_mean = reviews.points.mean()
reviews.points - review_points_mean
reviews.country + " - " + reviews.region_1
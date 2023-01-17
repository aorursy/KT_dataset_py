import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

pd.set_option('max_rows', 5)
reviews.head()
reviews.price.dtype
reviews.dtypes
reviews.info()
reviews.points.astype('float64')
reviews.index.dtype
reviews[reviews.country.isnull()]
reviews[reviews.taster_twitter_handle.isnull()]
reviews.region_2.fillna("Unknown")
reviews.region_2
reviews.taster_twitter_handle.replace("@kerinokeefe", "@kerino")
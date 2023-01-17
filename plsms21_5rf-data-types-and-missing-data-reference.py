import pandas as pd
reviews = pd.read_csv("../input/winemag-data-130k-v2.csv", index_col=0)
pd.set_option('max_rows', 5)
reviews.price.dtype
reviews.dtypes
reviews.points.astype('float64')
reviews.index.dtype
reviews[reviews.country.isnull()]
reviews.region_2.fillna("Unknown")
reviews.taster_twitter_handle.replace("@kerinokeefe", "@kerino")
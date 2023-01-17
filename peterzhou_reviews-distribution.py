import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from matplotlib import pyplot as plt
%matplotlib inline
rd = pd.read_csv("../input/reviews_detail.csv")
rd.head()
review_distribution = rd[["listing_id","id"]].groupby("listing_id").count()
plt.figure(figsize=(30,30))
plt.xticks(range(0,500,10))
plt.hist(review_distribution.id,bins=50)
review_distribution.describe()
ls = pd.read_csv("../input/listings_summary.csv")
ls.head(10)
ls[["id","price","minimum_nights","number_of_reviews","reviews_per_month","calculated_host_listings_count","availability_365"]].describe()
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from matplotlib import pyplot as plt
%matplotlib inline
reviews = pd.read_csv('../input/reviews_detail.csv')
#Reviews By Month
reviews["month"] = reviews.date.apply(lambda x:x[:7])
month_review = reviews[["listing_id","month"]].groupby("month").count()
plt.figure(figsize=(30,30))
plt.title("Monthly Reviews Pattern 2009-2017 ")
plt.bar(range(len(month_review.listing_id.values)),month_review.listing_id.values)
#"Monthly Reviews Pattern in each year 2011-2016
plt.figure(figsize=(20,20))
for i,year in enumerate([2011,2012,2013,2014,2015,2016]):
    temp = month_review[month_review.index >= str(year)]
    temp = temp[temp.index <= str(year+1)]
    plt.subplot(321+i)
    plt.plot(temp,linewidth=3)
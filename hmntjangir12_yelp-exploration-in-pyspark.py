# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# import pyspark to work on spark
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('yelp_data_exploration').getOrCreate()
# Any results you write to the current directory are saved as output.
#import yelp review data into dataframe
yelp_review = spark.read.json('../input/yelp_academic_dataset_review.json')
# import yelp business data into dataframe
yelp_business = spark.read.json('../input/yelp_academic_dataset_business.json')
# import yelp user data into dataframe
yelp_user = spark.read.json('../input/yelp_academic_dataset_user.json')
# import yelp tip data into dataframe
#yelp_tip = spark.read.json('../input/yelp_academic_dataset_tip.json')
# import yelp checkin data into dataframe
#yelp_checkin = spark.read.json('../input/yelp_academic_dataset_checkin.json')
# now see top most reviewed business.
# so take review data which has rating(stars) more than 3
review_star_three = yelp_review.filter('stars >3')
grouped_review = review_star_three.groupby('business_id').count()
review_sort = grouped_review.sort('count',ascending=False)
business_only = yelp_business.select('business_id','name','categories')
review_business_name = business_only.join(review_sort,'business_id','inner')
Top_ten_reviewed_business = review_business_name.limit(10)
Top_ten_reviewed_business.show()
from pyspark.sql.functions import split,explode
category = yelp_business.select('categories')
individual_category = category.select(explode(split('categories', ',')).alias('category'))
grouped_category = individual_category.groupby('category').count()
top_category = grouped_category.sort('count',ascending=False)
top_category.show(10,truncate=False)
rating = yelp_business.select('stars')
group_rating = rating.groupby('stars').count()
rating_top = group_rating.sort('count',ascending=False)
rating_top.show(truncate=False)
locations = yelp_business.select('business_id','city')
review_city = yelp_review.select('business_id')
merge_city = locations.join(review_city,'business_id','inner')
grouped_review_city = merge_city.groupby('city').count()
most_reviewed_city = grouped_review_city.groupby('city').sum()
most_reviewed_city.sort('sum(count)',ascending=False).show(10)

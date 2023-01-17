# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
users = pd.read_csv('../input/yelp_user.csv')[["user_id","name", "review_count", "average_stars"]]
users.head(15)

reviews = pd.read_csv('../input/yelp_review.csv')
reviews.head(15)
businesses = pd.read_csv('../input/yelp_business.csv')[["business_id", "categories"]]
businesses.head(20)
def bucket_for_user(user):
    if user['average_stars'] > 3.5:
        return 1
    if user['average_stars'] < 2.5:
        return 3
    return 2
users.head(10)
reviews.head(15)
users['pizza_bucket'] = users.apply(lambda user: bucket_for_user(user), axis=1)
users.sample(30)
#get all reviews for business_id
def get_reviews(review_table, business_id):
    return review_table.loc[reviews["business_id"]==business_id]
reviews_filter = reviews.set_index('user_id').join(users.set_index('user_id'), on='user_id', rsuffix = '_user').reset_index()
#reviews_filter = reviews.set_index('business_id')

reviews_filter = reviews_filter.set_index('business_id').join(businesses.set_index('business_id'), on='business_id', rsuffix= 'business_id').reset_index()
reviews_filter.head(10)
#table that shows group of people that reviewed the same business
business_id_of_interest = "z8oIoCT1cXz7gZP5GeU5OA"  # obtained from endpoint call
same_business = get_reviews(reviews_filter, business_id_of_interest)
same_bucket_in_business = same_business.loc[same_business["pizza_bucket"]==1.0]
#change this to a range of values being similar instead of exact same review
same_review_in_bucket = same_bucket_in_business[same_bucket_in_business["stars"]==4]
category_list = businesses[businesses["business_id"] == business_id_of_interest]["categories"].values[0].split(";")
category_list
same_review_in_bucket
#table that shows person that reviewed same restaurant and I
# random_user = "LkKOYOOxKDIDzkiQ5FfJEg"
def get_similar_reviews(similar_user, category_of_original_reviewed_business="Restaurants"):
    reviews_for_user = reviews_filter[reviews_filter["user_id"]==similar_user]
    reviews_for_user[reviews_for_user["stars"] > 3]
    reviews = set(reviews_for_user[reviews_for_user['categories'].str.contains(category_of_original_reviewed_business)]["business_id"].values)
    return reviews

#filter businesses by category and same stars
# Get a single user who reviewed the same place with the same score and is in same bucket
category_of_interest = category_list[0]
businesses_you_would_like = list(map(lambda user: get_similar_reviews(user, category_of_interest), same_review_in_bucket.sample(5)["user_id"].values.tolist()))
print("This is a list of five sets", businesses_you_would_like)
businesses_you_would_like = set().union(*businesses_you_would_like)
print("This is a set of all the businesses in each set", businesses_you_would_like)

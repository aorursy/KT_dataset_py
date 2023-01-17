# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# styles for seaborn
sns.set(style="ticks", palette="muted", color_codes=True)
sns.set_color_codes("pastel")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# import the dataset which contains ratings (Label) and reviews (Review) grouped by Course ID
data = pd.read_csv("../input/reviews_by_course.csv")
data.head()
# inspect dataset for missing values
print(data.info())
# fill NaN with empty string
data = data.fillna("")
# inspect dataset again
print(data.info())
# unique course ID
len(data["CourseId"].unique())
# find most popular courses
reviews_number = data["CourseId"].value_counts()
# show top20 courses by the reviews number
print(reviews_number[:20])
# plot barplot
sns.barplot(y=reviews_number[:20].index, x=reviews_number[:20],color="b").set(xlabel="number of ratings", ylabel="Course ID")
# average rating of the course
average_rating = data.groupby("CourseId").mean().sort_values("Label", ascending=False)
# plot histogram
sns.distplot(average_rating, kde=False).set(xlabel="average rating", ylabel="number of courses")
# plot histogram for average rating >= 3.8
sns.distplot(average_rating[average_rating.Label >= 3.8], kde=False).set(xlabel="average rating", ylabel="number of courses",)
# number of courses with average rating 5.0
av_rating_5 = int(average_rating[average_rating.Label==5.0].count())
print("Number of courses with average rating 5.0:")
print(av_rating_5)
print("% of the total number of courses:")
print(av_rating_5/len(data["CourseId"].unique())*100)
# extract review numbers for courses with average rating 5.0
df_av_rating_5 = pd.DataFrame(reviews_number)[average_rating.Label==5.0]
# explore statistics
print(df_av_rating_5.describe())
# compute the number of characters in the review
data["Review_len"] = data["Review"].str.len()
# compute the average number of characters in the review for every course
average_len = data.groupby("CourseId").mean().sort_values("Review_len", ascending=False)
# explore statistics
print(average_len.Review_len.describe())
#plot histogram
sns.distplot(average_len["Review_len"], kde=False).set(xlabel="average number of characters in review", ylabel="number of courses",)
# Merge datasets on index and create new DataFrame "analysis"
# first we will transform most_reviews from Series to DataFrame
df_reviews_number = pd.DataFrame(reviews_number)
# merge df_reviews_number and average_rating
analysis = pd.merge(df_reviews_number, average_rating,  right_index=True, left_index=True)
# transform average_len from Series to DataFrame
df_average_len = pd.DataFrame(average_len.Review_len)
# merge analysis and df_average_len
analysis = pd.merge(analysis, df_average_len,  right_index=True, left_index=True)
# rename columns
analysis.columns = ["reviews_number", "av_rating", "av_review_len"]
# show first 5 rows
analysis.head()
# Let's first explore correlations 
analysis.corr()
# plot scatter plot with av_review_len as x and av_rating as y
sns.scatterplot(x="av_review_len", y="av_rating", data = analysis).set(xlim=(0,500), xlabel="average number of characters in review", ylabel="average rating")
# Show scatter plot with linear model
sns.lmplot(x="av_review_len", y="av_rating", data = analysis).set(xlim=(-30, 500), ylim=(2,5.1), xlabel="average number of characters in review", ylabel="average rating")
# Count the number of "!" in the reviews
data["excl_num"] = data["Review"].str.count("!")
# Explote the statistics
print(data["excl_num"].describe())
print("Course Id:")
print(data.iloc[data["excl_num"].idxmax()]["CourseId"])
print("Review:")
print(data.iloc[data["excl_num"].idxmax()]["Review"])
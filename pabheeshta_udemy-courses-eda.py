%matplotlib inline

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import datetime as dt

sns.set(style = "whitegrid", font = "sans-serif", palette = "Dark2", font_scale = 1.1)
udemy_DF = pd.read_csv('../input/udemy-courses/udemy_courses.csv')
udemy_DF.shape
udemy_DF.head(10)
udemy_DF.isnull().sum()
udemy_DF[udemy_DF['num_lectures'] == 0]
udemy_DF.drop(udemy_DF[udemy_DF['num_lectures'] == 0].index, inplace = True)
udemy_DF['published_timestamp'] = pd.to_datetime(udemy_DF['published_timestamp'])

udemy_DF['published_date'] = udemy_DF['published_timestamp'].dt.date

udemy_DF['published_year'] = pd.DatetimeIndex(udemy_DF['published_date']).year
udemy_DF.head()
plt.figure(figsize = (9,4))

sns.countplot(data = udemy_DF, x = 'subject')

plt.show()
plt.figure(figsize = (9,4))

sns.countplot(data = udemy_DF, x = 'level')

plt.show()
plt.figure(figsize = (9,4))

sns.countplot(data = udemy_DF, x = 'is_paid')

plt.show()
plt.figure(figsize = (9,4))

sns.countplot(data = udemy_DF, x = 'published_year')

plt.show()
udemy_DF.nlargest(5, 'published_timestamp')
plt.figure(figsize = (10,4))

sns.distplot(udemy_DF['price'], color = "#4c84f5", kde = False)

plt.show()
plt.figure(figsize = (10,4))

sns.boxplot(udemy_DF['price'], color = "#4c84f5", linewidth = 2.2)

plt.show()
plt.figure(figsize = (8,4))

sns.distplot(udemy_DF['num_subscribers'], color = "#c92c26", kde = False)

plt.show()
plt.figure(figsize = (10,4))

sns.boxplot(udemy_DF['num_subscribers'], color = "#359c91", linewidth = 2.2)

plt.show()
plt.figure(figsize = (8,4))

sns.distplot(udemy_DF['num_reviews'], color = "#359c91", kde = False)

plt.show()
plt.figure(figsize = (10,4))

sns.boxplot(udemy_DF['num_reviews'], color = "#359c91", linewidth = 2.2)

plt.show()
udemy_DF['num_reviews'].describe()
plt.figure(figsize = (8,4))

sns.distplot(udemy_DF['num_lectures'], color = "#94d111", kde = False)

plt.show()
plt.figure(figsize = (10,4))

sns.boxplot(udemy_DF['num_lectures'], color = "#94d111", linewidth = 2.2)

plt.show()
udemy_DF['num_lectures'].describe()
plt.figure(figsize = (8,4))

sns.distplot(udemy_DF['content_duration'], color = "#ff03ea", kde = False)

plt.show()
plt.figure(figsize = (10,4))

sns.boxplot(udemy_DF['content_duration'], color = "#ff03ea", linewidth = 2.2)

plt.show()
udemy_DF['content_duration'].describe()
udemy_DF.nlargest(10, 'num_subscribers')
plt.figure(figsize = (10, 4))

sns.barplot(data = udemy_DF.nlargest(10, 'num_subscribers'), 

            x = 'num_subscribers', y = 'course_title')

plt.title(label = "Top 10 Most Subscribed Courses")

plt.show()
udemy_DF[udemy_DF['is_paid'] == True].nlargest(10, 'num_subscribers')
plt.figure(figsize = (10, 4))

sns.barplot(data = udemy_DF[udemy_DF['is_paid'] == True].nlargest(10, 'num_subscribers'), 

            x = 'num_subscribers', y = 'course_title')

plt.title(label = "Top 10 Most Subscribed Paid Courses")

plt.show()
udemy_DF[udemy_DF['is_paid'] == False].nlargest(10, 'num_subscribers')
plt.figure(figsize = (10, 4))

sns.barplot(data = udemy_DF[udemy_DF['is_paid'] == False].nlargest(10, 'num_subscribers'), 

            x = 'num_subscribers', y = 'course_title')

plt.title(label = "Top 10 Most Subscribed Free Courses")

plt.show()
plt.figure(figsize = (10,3))

sns.barplot(data = udemy_DF[udemy_DF['subject'] == 'Business Finance'].nlargest(10, 'num_subscribers'), 

            x = 'num_subscribers', y = 'course_title')

plt.title("Top 10 Most Subscribed Business Finance Courses")

plt.show()



plt.figure(figsize = (10,3))

sns.barplot(data = udemy_DF[udemy_DF['subject'] == 'Web Development'].nlargest(10, 'num_subscribers'), 

            x = 'num_subscribers', y = 'course_title')

plt.title("Top 10 Most Subscribed Web Development Courses")

plt.show()



plt.figure(figsize = (10,3))

sns.barplot(data = udemy_DF[udemy_DF['subject'] == 'Musical Instruments'].nlargest(10, 'num_subscribers'), 

            x = 'num_subscribers', y = 'course_title')

plt.title("Top 10 Most Subscribed Musical Instruments Courses")

plt.show()



plt.figure(figsize = (10,3))

sns.barplot(data = udemy_DF[udemy_DF['subject'] == 'Graphic Design'].nlargest(10, 'num_subscribers'), 

            x = 'num_subscribers', y = 'course_title')

plt.title("Top 10 Most Subscribed Graphic Design Courses")

plt.show()
udemy_DF.nlargest(10, 'num_reviews')
plt.figure(figsize = (10, 4))

sns.barplot(data = udemy_DF.nlargest(10, 'num_reviews'), 

            x = 'num_reviews', y = 'course_title')

plt.title(label = "Top 10 Most Reviewed Courses")

plt.show()
udemy_DF[udemy_DF['is_paid'] == True].nlargest(10, 'num_reviews')
plt.figure(figsize = (10, 4))

sns.barplot(data = udemy_DF[udemy_DF['is_paid'] == True].nlargest(10, 'num_reviews'), 

            x = 'num_reviews', y = 'course_title')

plt.title(label = "Top 10 Most Reviewed Paid Courses")

plt.show()
udemy_DF[udemy_DF['is_paid'] == False].nlargest(10, 'num_reviews')
plt.figure(figsize = (10, 4))

sns.barplot(data = udemy_DF[udemy_DF['is_paid'] == False].nlargest(10, 'num_reviews'), 

            x = 'num_reviews', y = 'course_title')

plt.title(label = "Top 10 Most Reviewed Free Courses")

plt.show()
plt.figure(figsize = (10,3))

sns.barplot(data = udemy_DF[udemy_DF['subject'] == 'Business Finance'].nlargest(10, 'num_reviews'), 

            x = 'num_reviews', y = 'course_title')

plt.title("Top 10 Most Reviewed Business Finance Courses")

plt.show()



plt.figure(figsize = (10,3))

sns.barplot(data = udemy_DF[udemy_DF['subject'] == 'Web Development'].nlargest(10, 'num_reviews'), 

            x = 'num_reviews', y = 'course_title')

plt.title("Top 10 Most Reviewed Web Development Courses")

plt.show()



plt.figure(figsize = (10,3))

sns.barplot(data = udemy_DF[udemy_DF['subject'] == 'Musical Instruments'].nlargest(10, 'num_reviews'), 

            x = 'num_reviews', y = 'course_title')

plt.title("Top 10 Most Reviewed Musical Instruments Courses")

plt.show()



plt.figure(figsize = (10,3))

sns.barplot(data = udemy_DF[udemy_DF['subject'] == 'Graphic Design'].nlargest(10, 'num_reviews'), 

            x = 'num_reviews', y = 'course_title')

plt.title("Top 10 Most Reviewed Graphic Design Courses")

plt.show()

plt.figure(figsize = (12,4))

sns.countplot(data = udemy_DF, x = 'subject', hue = 'is_paid')

plt.show()
plt.figure(figsize = (12,4))

sns.countplot(data = udemy_DF, x = 'subject', hue = 'level')

plt.show()
plt.figure(figsize = (18,4))

sns.countplot(data = udemy_DF, x = 'subject', hue = 'published_year')

plt.show()
plt.figure(figsize=(10, 5))

sns.boxplot(data = udemy_DF, x = 'price', y = 'subject', linewidth = 2.2)

plt.show()
udemy_DF.groupby(['subject']).describe()['price']
plt.figure(figsize=(10, 5))

sns.boxplot(data = udemy_DF, x = 'num_subscribers', y = 'subject', linewidth = 2.2, showfliers = True)

plt.show()
udemy_DF.groupby(['subject']).describe()['num_subscribers']
plt.figure(figsize=(10, 5))

sns.boxplot(data = udemy_DF, x = 'num_reviews', y = 'subject', linewidth = 2.2, showfliers = True)

plt.show()
udemy_DF.groupby(['subject']).describe()['num_reviews']
plt.figure(figsize=(10, 5))

sns.boxplot(data = udemy_DF, x = 'content_duration', y = 'subject', linewidth = 2.2, showfliers = True)

plt.show()
udemy_DF.groupby(['subject']).describe()['content_duration']
udemy_corr = udemy_DF[['price', 'num_subscribers', 'num_reviews', 'num_lectures', 'content_duration', 'published_year']].corr()

plt.figure(figsize=(8,8))

sns.heatmap(udemy_corr, annot = True, linewidths = 1.2, linecolor = 'white')

plt.xticks(rotation = 75)

plt.show()
udemy_DF_no_subs = udemy_DF[udemy_DF['num_subscribers'] == 0]

udemy_DF_no_revs = udemy_DF[udemy_DF['num_reviews'] == 0]
udemy_DF_no_subs[['num_subscribers', 'num_reviews', 'num_lectures', 'content_duration']].describe()
udemy_DF_no_revs[['num_subscribers', 'num_reviews', 'num_lectures', 'content_duration']].describe()
udemy_DF_no_revs[udemy_DF_no_revs['num_subscribers'] > 0].nlargest(5, 'num_subscribers')
f, ax1 = plt.subplots(4, 2, figsize = (20, 20))

sns.countplot(data = udemy_DF_no_subs, x = "is_paid", ax = ax1[0,0])

sns.countplot(data = udemy_DF_no_revs, x = "is_paid", ax = ax1[0,1])

sns.countplot(data = udemy_DF_no_subs, x = "level", ax = ax1[1,0])

sns.countplot(data = udemy_DF_no_revs, x = "level", ax = ax1[1,1])

sns.countplot(data = udemy_DF_no_subs, x = "subject", ax = ax1[2,0])

sns.countplot(data = udemy_DF_no_revs, x = "subject", ax = ax1[2,1])

sns.countplot(data = udemy_DF_no_subs, x = "published_year", ax = ax1[3,0])

sns.countplot(data = udemy_DF_no_revs, x = "published_year", ax = ax1[3,1])

plt.show()
udemy_DF_no_revs.groupby(['published_year']).count()
f, ax2 = plt.subplots(3, 2, figsize = (20, 20))

sns.distplot(udemy_DF_no_subs['price'], ax = ax2[0,0], kde = False, bins = 20)

sns.distplot(udemy_DF_no_revs['price'], ax = ax2[0,1], kde = False, bins = 20)

sns.distplot(udemy_DF_no_subs['num_lectures'], ax = ax2[1,0], kde = False)

sns.distplot(udemy_DF_no_revs['num_lectures'], ax = ax2[1,1], kde = False)

sns.distplot(udemy_DF_no_subs['content_duration'], ax = ax2[2,0], kde = False)

sns.distplot(udemy_DF_no_revs['content_duration'], ax = ax2[2,1], kde = False, bins = 10)

plt.show()
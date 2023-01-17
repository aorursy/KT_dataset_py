##Author: Durbanova Anna

## Date: 09.08.2020



import pandas as pd

import os

import glob

import numpy as np

import holoviews as hv

import seaborn as sns

import matplotlib.pyplot as plt

from datetime import datetime

hv.extension('bokeh')

%matplotlib inline







#import pandas as pd

#from datetime import datetime

#import numpy as np

#import hvplot

#from hvplot import hvPlot

#import hvplot.pandas

#import seaborn as sns

#import matplotlib.pyplot as plt

#import  pingouin as pg

#%matplotlib inline

#!pip install hvplot

import hvplot

from hvplot import hvPlot

import hvplot.pandas
#!pip install pingouin
import pingouin as pg
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

data=pd.read_csv('/kaggle/input/udemy-courses/udemy_courses.csv')





data["published_timestamp"]=pd.to_datetime(data.published_timestamp, format="%Y-%m-%dT%H:%M:%SZ")

data
## What is the most popular prices that the majority of courses have?

ratio=(data

 .groupby("price")

 [["course_title"]]

 .count()

 .sort_values("course_title", ascending=False)

)

ratio["percentage"]=ratio["course_title"]/ratio["course_title"].sum()

## Note: 22.5% of all courses have a price of 20€, 

## 12,7% of courses have 50€ price,

## 8% of courses are FREE, 

## and also 8 % of courses have 200€ Price



## The least of courses have the price more than 100€ 



fig,(ax1, ax2)= plt.subplots(ncols=2, figsize=[12,4])

sns.kdeplot(data["price"], data["course_id"], shade=True, ax=ax1).set_title("Price Range for Udemy Courses");

sns.kdeplot(data["price"], shade=True, ax=ax2).set_title("Price Range for Udemy Courses");
sns.jointplot(data["price"], data["course_id"], kind="kde");
data["content_duration"].value_counts().head(15)
fig, (ax1,ax2)=plt.subplots(ncols=2, figsize=[12,4])

sns.kdeplot(data["content_duration"], data["course_id"], shade=True, ax=ax1);

sns.kdeplot(data["content_duration"], shade=True, ax=ax2);

## Notes: Most of the courses have a duration of less than 10 hours. Mostly in the range between 1-5 hours
## Price Range and Duration

#fig, (ax1, ax2)=plt.subplots(ncols=2, figsize=[12,4])

sns.kdeplot(data["content_duration"], data["price"], shade=True);
#data.num_lectures.median()

sns.distplot(data.num_lectures); 

#Data distribution is not bell-shaped, so let's use the median for calculating the average of the content duration
data.num_lectures.median()



## On average, every Udemy course has 25 lectures
(data

 .groupby(data.num_lectures)

 [["num_subscribers"]]

 .median()

 .sort_values("num_subscribers", ascending=False)

 .head(10)

)

sns.set(style="whitegrid")

sns.catplot(x="num_lectures", y="num_subscribers", 

                capsize=.2, palette="YlGnBu_d", height=6, aspect=.75,

                kind="point", data=data);

## Note: It does not really matter, if the are more lectures, there are more subscribers

(data

.groupby(data.subject)

 [["course_title"]]

 .count()

 .sort_values("course_title", ascending=False)

)



## Note: Most of the courses are devoted to Web Development and Business Finance Topics

data.hvplot.bar(x="subject", y="num_subscribers", rot=90)

#However there are more subscribers for WebDevelopment and Musical Instruments
data.level.value_counts()

## Note:Most of the courses are for All levels applicable,

##If we look at separate levels, there are more courses for beginners
(data

.groupby("level")

 [["num_subscribers"]]

 .sum()

 .sort_values("num_subscribers", ascending=False)

)

#In general Courses with "All Levels" and Beginner Level has the most subscribers

(data

 .groupby("price")

 [["num_subscribers"]]

 .sum()

 .sort_values("num_subscribers", ascending=False)

 .head()

)

# Most of subscribers are of course for FREE courses

# 1,3 miliion subscribers are of courses with 200€ and 20€

data.pivot_table(values="num_subscribers",index="price", columns="subject", aggfunc="sum").style.background_gradient()
## Note: In general, the most subscribers are for FREE courses, whenever the subject is. 

## The second place of popularity are couses with 200€
## Let's look on average, how many subscribers would be on a given day

data.pivot_table(values="num_subscribers",index="price", columns="subject", aggfunc="median").style.background_gradient()



## For Graphic Design Courses we can see a gradual increase of interest with an increased price. 

## For Business Finance and Web Development some courses with the price 155 or 185 have also an increased number of subscribers
(data

 .groupby("subject")

 [["num_subscribers", "num_reviews"]]

 .sum()

 .sort_values("num_reviews", ascending= False)

).style.background_gradient()



## More reviews and subscribers for Web Development

## Less on Musical Instruments

(data

.groupby("subject")

 [["num_subscribers", "num_lectures", "content_duration"]]

 .sum()

 .sort_values(["num_lectures", "content_duration"], ascending=False)

).style.background_gradient()





## Web Dvelopment and Business Finacne courses have the most lectures in total and the longest content duration

## 
plt.rcParams["figure.figsize"]=(12,10)

sns.heatmap(data.corr(), cmap="BuPu", annot=True);

## Most correlation is observed between Number of lectures and Content Duration

## Another correlation is between Number of Subscribers and Number of Reviews

## Price and Free or not free has a slight correlation. 
data.corr().style.background_gradient()
query = "num_reviews < 600 & num_subscribers < 5000"

sns.jointplot(x="num_reviews", y="num_subscribers", kind="scatter", data=data.query(query), s=1.8);
pg.corr(data["num_reviews"], data["num_subscribers"])

## This correlation is statistically significant
pg.corr(data["content_duration"], data["num_lectures"])

## This correlation is statistically significant
data["num_subscribers"].describe()
data[data["num_subscribers"]==268923]

## The course for Web Development "Learn HTML 5 Programming from Scratch " is one of the popular on Udemy

## It is free,  
mask= data["num_subscribers"]==0

(data[mask]

.groupby("course_title")

 ["price", "num_subscribers", "num_reviews", "num_lectures", "content_duration"]

.sum()

.head(75)

)



## There are 65 courses without subscribers 

## Analyzing data of the least popular courses

data[mask].describe()

## On average the price is about 45€ without subscribers and reviews 

## Btw the number of reviews, have a correlation with number of subscribers

## On average there are 13 lectures per course with a duration of 1.49 on average
## When do usually the courses are published, on which days and day of weeks



(data

.pivot_table(

    values="course_id", 

    aggfunc=len, 

    index=data.published_timestamp.dt.month_name(),

    columns=data.published_timestamp.dt.day_name())

 .style.background_gradient()

)
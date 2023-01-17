# Importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
hn=pd.read_csv('/kaggle/input/hacker-news-posts/HN_posts_year_to_Sep_26_2016.csv', low_memory=True)

hn.head()
hn.info()
hn.author.value_counts().head(10)
print('Data science'.startswith('data'))

print('data science'.startswith('data'))
# creating empty lists to store the filtered data in

ask_posts=[]

show_posts=[]

other_posts=[]
for i,x in hn.iterrows():

    if x.title.lower().startswith('ask hn'):

        ask_posts.append(x)

    elif x.title.lower().startswith('show hn'):

        show_posts.append(x)

    else:

        other_posts.append(x)
print ('Number of Ask HN posts are:', len(ask_posts))

print ('Number of Show HN posts are:', len(show_posts))

print ('Number of Other posts are:', len(other_posts))
ask_posts[:5]
total_ask_comments=0

for i in ask_posts:

    total_ask_comments+= i[4]

avg_ask_comments= total_ask_comments/ len(ask_posts)

print ('Total comments on Ask HN posts are:', total_ask_comments)

print ('Average comments on Ask HN posts are:', avg_ask_comments)
total_show_comments=0

for i in show_posts:

    total_show_comments+= i[4]

avg_show_comments= total_show_comments/ len(show_posts)

print ('Total comments on Show HN posts are:', total_show_comments)

print ('Average comments on Show HN posts are:', avg_show_comments)
import datetime as dt

result_list=[]

for x in ask_posts:

    result_list.append([x[6], x[4]])

result_list[:5]
counts_by_hour={}

comment_by_hour={}

for i in result_list:

    x=dt.datetime.strptime(i[0], "%m/%d/%Y %H:%M")

    hour=dt.datetime.strftime(x, '%H')

    if hour in counts_by_hour.keys():

        counts_by_hour[hour]+=1

        comment_by_hour[hour]+= i[1]

    else:

        counts_by_hour[hour]=1

        comment_by_hour[hour]=i[1]
counts_by_hour
avg_by_hour=[]

for comment in comment_by_hour:

        avg_by_hour.append([comment, round(comment_by_hour[comment]/counts_by_hour[comment], 2)])
print('The average number of comments per post are:')

sorted(avg_by_hour)
swap_avg_by_hour=[[value, key] for key, value in avg_by_hour]
sorted_swap=sorted(swap_avg_by_hour, reverse=True)
print("Top 5 Hours for Ask Posts Comments")

for i in sorted_swap[:5]:

    time=dt.datetime.strptime(i[1], '%H')

    hour= dt.datetime.strftime(time,'%H:%M')

    print(hour, i[0])
total_ask_votes=0

for i in ask_posts:

    total_ask_votes+= i[3]

avg_ask_votes= total_ask_votes/ len(ask_posts)

print ('Total votes on Ask HN posts are:', total_ask_votes)

print ('Average votes on Ask HN posts are:', avg_ask_votes)
total_show_votes=0

for i in show_posts:

    total_show_votes+= i[3]

avg_ask_votes= total_show_votes/ len(ask_posts)

print ('Total votes on Show HN posts are:', total_show_votes)

print ('Average votes on Show HN posts are:', avg_ask_votes)
import datetime as dt

result_list=[]

for x in ask_posts:

    result_list.append([x[6], x[3]])

print(result_list[:5])



## Converting to hour

count_by_hour={}

vote_by_hour={}

for i in result_list:

    time=dt.datetime.strptime(i[0], '%m/%d/%Y %H:%M')

    hour=dt.datetime.strftime(time, '%H')

    if hour in count_by_hour.keys():

        count_by_hour[hour]+=1

        vote_by_hour[hour]+=i[1]

    else:

        count_by_hour[hour]=1

        vote_by_hour[hour]=i[1]
vote_by_hour
avgvote_by_hour=[]

for vote in vote_by_hour:

        avgvote_by_hour.append([vote, round(vote_by_hour[vote]/count_by_hour[vote], 2)])



sort_list=[[value, key] for key, value in avgvote_by_hour ]

sorted_swap=sorted(sort_list, reverse=True)
print("Top 5 Hours for Ask Posts Comments")

for i in sorted_swap[:5]:

    time=dt.datetime.strptime(i[1], '%H')

    hour= dt.datetime.strftime(time,'%H:%M')

    print(hour, i[0])
total_other_comments=0

total_other_votes=0

for i in other_posts:

    total_other_comments+= i[4]

    total_other_votes+=i[3]

avg_other_comments= total_other_comments/ len(other_posts)

avg_other_votes=total_other_votes/ len(other_posts)

print ('Average comments on Other posts are:', avg_other_comments)

print ('Average votes on Other HN posts are:', avg_other_votes)
total_ask_votes=0

total_show_votes=0

for i in ask_posts:

    total_ask_votes+=i[3]

for show in show_posts:

    total_show_votes+= show[3]

avg_ask_votes= total_ask_votes/ len(ask_posts)

avg_show_votes=total_show_votes/ len(show_posts)

print ('Average votes on Ask posts are:', avg_ask_votes)

print ('Average votes on Show HN posts are:', avg_show_votes)
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
# Read the data



import csv



file = open('../input/hacker-news-posts/hacker_news.csv')

hn = list(csv.reader(file))

print(hn[:5])
# Removing header



header = hn[0]

hn = hn[1:]

print(header)

print('\n', hn[:5])
# Creating empty list



ask_posts = [] # For Ask HN posts

show_posts = [] # For Show HN posts

other_posts = [] # For other posts



# Creating a loop over the data to separate the posts



for post in hn:

    title = post[1] # Since title is second column

    title = title.lower() # Converting into lower-case

    if title.startswith('ask hn') is True:

        ask_posts.append(post)

    elif title.startswith('show hn') is True:

        show_posts.append(post)

    else:

        other_posts.append(post)

        

# Checking the number of posts



print('Number of Ask HN posts: ', len(ask_posts))

print('Number of Show HN posts: ', len(show_posts))

print('Number of other posts: ', len(other_posts))
# Calculating first for Ask HN posts



total_ask_comments = 0



for posts in ask_posts:

    num_comments = int(posts[4])

    total_ask_comments += num_comments

    

avg_ask_comments = total_ask_comments / len(ask_posts)



print('Average Number of Comments on Ask posts: ', avg_ask_comments)
# Calculating for Show HN posts



total_show_comments = 0



for posts in show_posts:

    num_comments = int(posts[4])

    total_show_comments += num_comments

    

avg_show_comments = total_show_comments / len(show_posts)



print('Average Number of Comments on Show posts: ', avg_show_comments)
# Calculating the amount of ask posts created in each hour of the day, along with the number of comments received.



import datetime as dt



result_list = [] # To store the results



for post in ask_posts:

    created_at = post[6]

    num_comments = int(post[4])

    result_list.append([created_at, num_comments])

    

counts_by_hour = {}

comments_by_hour = {}



for each_row in result_list:

    date = each_row[0]

    comment = each_row[1]

    date = dt.datetime.strptime(date, "%m/%d/%Y %H:%M")

    time = date.strftime("%H")

    if time not in counts_by_hour:

        counts_by_hour[time] = 1

        comments_by_hour[time] = comment

    else:

        counts_by_hour[time] += 1

        comments_by_hour[time] += comment

        

comments_by_hour
avg_by_hour = []



for hr in comments_by_hour:

    avg_by_hour.append([hr, round(comments_by_hour[hr] / counts_by_hour[hr],3)])



avg_by_hour
swap_avg_by_hour = []



for hr in avg_by_hour:

    swap_avg_by_hour.append([hr[1],hr[0]])



swap_avg_by_hour
sorted_swap = sorted(swap_avg_by_hour, reverse=True)



print("Top 5 Hours for Ask Posts Comments")



for avg, hr in sorted_swap[:5]:

    print("{}: {:.2f} average comments per post".format(

        dt.datetime.strptime(hr, "%H").strftime("%H:%M"),avg)

         )



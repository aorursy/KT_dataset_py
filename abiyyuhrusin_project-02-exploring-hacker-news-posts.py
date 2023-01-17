# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from csv import reader

opened_file = open("/kaggle/input/hacker-news-posts/hacker_news.csv")

read_file = reader(opened_file)

hn = list(read_file)

hn[:5]
headers = hn[0]

headers
del hn[0]
hn[:5]
ask_posts = []

show_posts = []

other_posts = []



for i in hn:

    title = i[1]

    if title.lower().startswith('ask hn'):

        ask_posts.append(i)

    elif title.lower().startswith('show hn'):

        show_posts.append(i)

    else:

        other_posts.append(i)
print(ask_posts[:5])

print(len(ask_posts))

print('\n')

print(show_posts[:5])

print(len(show_posts))

print('\n')

print(other_posts[:5])

print(len(other_posts))
total_ask_comment = 0

for i in ask_posts:

    cmnt = i[4]

    total_ask_comment += float(i[4])

    

avg_ask_comments = total_ask_comment / len(ask_posts)

print(avg_ask_comments)

    
total_show_comment = 0

for i in show_posts:

    cmnt = i[4]

    total_show_comment += float(i[4])

    

avg_show_comments = total_show_comment / len(show_posts)

print(avg_show_comments)
import datetime as dt

result_list = []



for i in ask_posts:

    d1 = i[6]

    ncmnt = i[4]

    result_list.append([d1,ncmnt])
result_list
counts_by_hour = {}

comments_by_hour = {}



for i in result_list:

    date_dt = dt.datetime.strptime(i[0], "%m/%d/%Y %H:%M")

    

    

    if date_dt.hour not in counts_by_hour:

        counts_by_hour[date_dt.hour] = 1

        comments_by_hour[date_dt.hour] = float(i[1])

    else:

        counts_by_hour[date_dt.hour] += 1

        comments_by_hour[date_dt.hour] += float(i[1])
print(counts_by_hour)

print('\n')

print(comments_by_hour)
avg_by_hour = []



for i in comments_by_hour:

    avg_by_hour.append([i, comments_by_hour[i] / counts_by_hour[i]])



avg_by_hour
swap_avg_by_hour = []



for i in avg_by_hour:

    swap_avg_by_hour.append([i[1],i[0]])

    

swap_avg_by_hour
sorted_swap = sorted(swap_avg_by_hour, reverse=True)

sorted_swap
#15:00: 38.59 average comments per post



for i in sorted_swap:

    hour = dt.datetime.strptime(str(i[1]), '%H')

    print('{}: {:.2f} average comments per post'.format(hour.strftime('%H:%M'), i[0]))
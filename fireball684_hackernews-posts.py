import os



os.listdir('../input')
import csv 

with open('../input/HN_posts_year_to_Sep_26_2016.csv') as file:

    f_data = csv.reader(file)

    hn = list(f_data)
type(hn)
# First 5 rows of the dataset

# hn[:5]
headers = hn[0]
del hn[0]
headers
hn[:5]
ask_posts = []

show_posts = []

other_posts = []
title = []

for row in hn:

    title.append(row[1])
for row in hn:

    if row[1].lower().startswith('ask hn'):

        ask_posts.append(row)

    elif row[1].lower().startswith('show hn'):

        show_posts.append(row)

    else:

        other_posts.append(row)
len(ask_posts), len(show_posts), len(other_posts)
ask_posts[:2]
total_ask_comments  = 0
for row in ask_posts:

    total_ask_comments += int(row[4])

    

avg_ask_comments = total_ask_comments / len(ask_posts)

print(avg_ask_comments)
total_show_comments = 0
for row in show_posts:

    total_show_comments += int(row[4])

    

avg_show_comments = total_show_comments / len(show_posts)

print(avg_show_comments)
import datetime as dt



result_list = []

for row in ask_posts:

    result_list.append([row[6], int(row[4])])

    

counts_by_hour = {}

comments_by_hour = {}
result_list[:2]
for row in result_list:

    dt_obj = dt.datetime.strptime(row[0], '%m/%d/%Y %H:%M')

    dt_H = dt_obj.strftime('%H')

    if dt_H not in counts_by_hour:

        counts_by_hour[dt_H] = 1

        comments_by_hour[dt_H] = row[1]

    else:

        counts_by_hour[dt_H] += 1

        comments_by_hour[dt_H] += row[1]    
dt_obj.strftime('%H')
counts_by_hour.keys()
avg_by_hour = []

for key in counts_by_hour.keys():

    avg_by_hour.append([key, comments_by_hour[key]/counts_by_hour[key]])
avg_by_hour.sort(key= lambda x:int(x[0]))

print(avg_by_hour)
swap_avg_by_hour = []

for row in avg_by_hour:

    swap_avg_by_hour.append([row[1], row[0]])
sorted_swap = sorted(swap_avg_by_hour, reverse=True)
print('Top 5 Hours for Ask Posts Comments')

for row in sorted_swap[:5]:

    print('{}:00: {:.2f} average comments per post.'.format(row[1], row[0]))
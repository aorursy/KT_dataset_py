#Reading all the required libraries

from csv import reader

import datetime as dt
opened_file = open('hacker_news.csv')

read_file = reader(opened_file)

hn = list(read_file)

hn[0:5]
headers = hn[0]
hn = hn[1:]
headers
hn[0:5]
ask_posts = []

show_posts = []

other_posts = []
for row in hn:

    title = row[1]

    title = title.lower()

    if title.startswith('ask hn'):

        ask_posts.append(row)

    elif title.startswith('show hn'):

        show_posts.append(row)

    else:

        other_posts.append(row)
print(len(ask_posts))

print(len(show_posts))

print(len(other_posts))
ask_posts[0:5]
show_posts[0:5]
total_ask_comments = 0

for row in ask_posts:

    comments = int(row[4])

    total_ask_comments += comments

    

avg_ask_comments = round(total_ask_comments/len(ask_posts))

print(avg_ask_comments)



total_show_comments = 0

for row in show_posts:

    comments = int(row[4])

    total_show_comments += comments

    

avg_show_comments = round(total_show_comments/len(show_posts))

print(avg_show_comments)
result_list = []



for post in ask_posts:

    result_list.append(

        [post[6], int(post[4])]

    )



print(result_list)
res
comments_by_hour = {}

counts_by_hour = {}

date_format = "%m/%d/%Y %H:%M"



for each_row in result_list:

    date = each_row[0]

    comment = each_row[1]

    time = dt.datetime.strptime(date, date_format).strftime("%H")

    if time in counts_by_hour:

        comments_by_hour[time] += comment

        counts_by_hour[time] += 1

    else:

        comments_by_hour[time] = comment

        counts_by_hour[time] = 1



print(counts_by_hour)

print(comments_by_hour)

    
avg_by_hour = []



for row in comments_by_hour:

    avg_by_hour.append([row,comments_by_hour[row]/counts_by_hour[row]])

    

avg_by_hour
len(avg_by_hour)
swap_avg_by_hours = []



for row in avg_by_hour:

    swap_avg_by_hours.append([row[1],row[0]])

    

print(swap_avg_by_hours)
sorted_swap = sorted(swap_avg_by_hours,reverse=True)
sorted_swap
sorted_swap[:5]
for avg,hr in sorted_swap[:5]:

    print("{}: {:.2f} avearge comments per post".format(dt.datetime.strptime(hr,"%H").strftime("%H:%M"),avg))

    

    
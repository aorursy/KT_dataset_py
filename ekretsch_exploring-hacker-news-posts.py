#opening the Hacker News Dataset

import csv



file = open('../input/hacker-news-posts/HN_posts_year_to_Sep_26_2016.csv')

hn = list(csv.reader(file))

print(hn[:5])
headers = hn[0]

hn = hn[1:]

print(headers)

print(hn[:5])
#Separating post data into different lists by Title

ask_posts = []

show_posts = []

other_posts = []



for post in hn:

    title = post[1]

    if title.lower().startswith('ask hn'):

        ask_posts.append(post)

    elif title.lower().startswith("show hn"):

        show_posts.append(post)

    else:

        other_posts.append(post)

        

print(len(ask_posts))

print(len(show_posts))

print(len(other_posts))
#Calculating the average number of comments 'Ask HN' posts receive.

total_ask_comments = 0



for post in ask_posts:

    total_ask_comments += int(post[4])

    

avg_ask_comments = total_ask_comments / len(ask_posts)

print(avg_ask_comments)
#Calculating the average number of comments 'Show HN' posts receive.

total_show_comments = 0



for post in show_posts:

    total_show_comments += int(post[4])

    

avg_show_comments = total_show_comments / len(show_posts)

print(avg_show_comments)
#Calculating the amount of ask posts created during each hour of the day and the number of comments received.

import datetime as dt



result_list = []



for post in ask_posts:

    result_list.append(

        [post[6], int(post[4])]

    )

    

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



comments_by_hour
# Calculating the average amount of comments `Ask HN` posts created at each hour of the day receive.

avg_by_hour = []



for hr in comments_by_hour:

    avg_by_hour.append([hr, comments_by_hour[hr] / counts_by_hour[hr]])



avg_by_hour
swap_avg_by_hour = []



for row in avg_by_hour:

    swap_avg_by_hour.append([row[1], row[0]])

    

print(swap_avg_by_hour)



sorted_swap = sorted(swap_avg_by_hour, reverse=True)



sorted_swap
# Sorting the values and printing the the 5 hours with the highest average comments.



print("Top 5 Hours for 'Ask HN' Comments")

for avg, hr in sorted_swap[:5]:

    print(

        "{}: {:.2f} average comments per post".format(

            dt.datetime.strptime(hr, "%H").strftime("%H:%M"),avg

        )

    )
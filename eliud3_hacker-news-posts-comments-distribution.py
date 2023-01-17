from csv import reader

dfile = ('../input/hacker-news-posts/HN_posts_year_to_Sep_26_2016.csv')

opened_file = open(dfile)

hn = reader(opened_file)

hn = list(hn)

print(hn[:5])
header = hn[0]

hn = hn[1:]

print(hn[:5])
ask_posts = []

show_posts = []

other_posts = []



for row in hn:

    title = row[1]

    if title.lower().startswith("ask hn"):

        ask_posts.append(row)

    elif title.lower().startswith("show hn"):

        show_posts.append(row)

    else:

        other_posts.append(row)

        

print(len(ask_posts))

print(len(show_posts))

print(len(other_posts))
total_ask_comments = 0



for row in ask_posts:

    counter = row[4]

    total_ask_comments += int(counter)

    

avg_ask_comments = total_ask_comments / len(ask_posts)

print(avg_ask_comments)
total_show_comments = 0



for row in show_posts:

    counter = row[4]

    total_show_comments += int(counter)

    

avg_show_comments = total_show_comments / len(show_posts)

print(avg_show_comments)
import datetime as dt



result_list = []



for post in ask_posts:

    created_time = post[6]

    n_comments = int(post[4])

    result_list.append([created_time, n_comments])

    

counts_by_hour = {} #number of ask posts

comments_by_hour = {} #corresponding number of comments in each ask post

date_format = "%m/%d/%Y %H:%M"



for row in result_list:

    date = row[0]

    comment = row[1]

    time = dt.datetime.strptime(date, date_format).strftime("%H")

    if time in counts_by_hour:

        comments_by_hour[time] += comment

        counts_by_hour[time] += 1

    else:

        comments_by_hour[time] = comment

        counts_by_hour[time] = 1



print(comments_by_hour)

print('\n')

print(counts_by_hour)
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
print("Top 5 Hours for 'Ask HN' Comments")

for avg, hr in sorted_swap[:5]:

    print(

        "{}: {:.2f} average comments per post".format(

            dt.datetime.strptime(hr, "%H").strftime("%H:%M"),avg

        )

    )
#opening the hacker news dataset

from csv import reader

opened_file = open('../input/hacker-news/hacker_news.csv')

read_file = reader(opened_file)

hn = list(read_file)

print(hn[:5])

#cleaning the data by removing the header row

headers = hn[0]

hn = hn[1:]

print("headers: ", headers, "\n")

print("First Five Rows: ", hn[:5])
#categorizing posts based on Ask and Show HN

ask_posts = []

show_posts = []

other_posts = []



for row in hn:

    title = row[1]

    if title.lower().startswith('ask hn'):

        ask_posts.append(row)

    elif title.lower().startswith('show hn'):

        show_posts.append(row)

    else:

        other_posts.append(row)

        

print("number of posts asking HN = ", len(ask_posts), "\n",

      "number of posts showing HN = ", len(show_posts), "\n",

      "number of other posts = ", len(other_posts), "\n",

     )
# Finding the total number of comments to 'ask posts'

total_ask_comments = 0

for post in ask_posts:

    num_comments = int(post[4])

    total_ask_comments += num_comments



#calculating the avergae number of comments on 'ask posts'

avg_ask_comments = total_ask_comments / len(ask_posts)

print("The average number of comments on ask posts is : ", avg_ask_comments)



# Finding the total number of comments to 'show posts'

total_show_comments = 0

for post in show_posts:

    num_comments = int(post[4])

    total_show_comments += num_comments



#calculating the avergae number of comments on 'show posts'

avg_show_comments = total_show_comments / len(show_posts)

print("The average number of comments on show posts is : ", avg_show_comments)

import datetime as dt

result_list = []

for post in ask_posts:

    created_at = post[6]

    num_comments = int(post[4])

    mixed_list = [created_at, num_comments]

    result_list.append(mixed_list)

    

counts_by_hour = {}

comments_by_hour ={}

for row in result_list:

    date = row[0]

    date_dt = dt.datetime.strptime(date, "%m/%d/%Y %H:%M")

    hour = date_dt.strftime("%H")

    if hour not in counts_by_hour:

        counts_by_hour[hour] = 1

        comments_by_hour[hour] = row[1]

    else:

        counts_by_hour[hour] += 1

        comments_by_hour[hour] += row[1]

        

print("counts by hour : ", counts_by_hour, "\n", "\n",

     "comments by hour : ", comments_by_hour)

        

    
#calculating the average number of comments per post per hour

avg_by_hour = []

for hour in comments_by_hour:

    avg_by_hour.append([hour, (comments_by_hour[hour] / counts_by_hour[hour])])

print("The average number of comments per post for each hour of the day is: \n \n ", 

     avg_by_hour)    
#swap the columns

swap_avg_by_hour = []

for hour in avg_by_hour:

    swap_avg_by_hour.append([hour[1], hour[0]])



print(swap_avg_by_hour)



#sort the results

sorted_swap = sorted(swap_avg_by_hour, reverse = True)



print('\n Top 5 Hours for Ask Posts Comments: \n')

for avg, hour in sorted_swap[:5]:

    hour_dt = hour

    hour_ob = dt.datetime.strptime(hour_dt, "%H")

    hour_sf = hour_ob.strftime("%H:%M")

    avg_comments = "{}: {:.2f} average comments per post."

    print(avg_comments.format(hour_sf, avg))

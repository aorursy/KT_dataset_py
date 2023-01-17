import csv

from datetime import datetime



# read in the data

file = open("../input/HN_posts_year_to_Sep_26_2016.csv")

data = csv.reader(file)

hn = list(data)



hn[0:5]
# Remove the headers

headers = hn[0]

hn = hn[1:]



print(headers)

print(hn[0:5])
# Identify posts that begin with either `Ask HN` or `Show HN` and separate the data into different lists.

ask_posts = []

show_posts = []

other_posts = []



for item in hn:

    title = item[1]

    if title.lower().startswith("ask hn"):

        ask_posts.append(item)

    elif title.lower().startswith("show hn"):

        show_posts.append(item)

    else:

        other_posts.append(item)

        

print(len(ask_posts))

print(len(show_posts))

print(len(other_posts))
# Calculate the average number of comments `Ask HN` posts receive.

total_ask_comments = 0

for item in ask_posts:

    total_ask_comments += int(item[4])

    

avg_ask_comments = total_ask_comments / len(ask_posts)

avg_ask_comments
# Calculate the average number of comments `Show HN` posts receive.

total_show_comments = 0

for item in show_posts:

    total_show_comments += int(item[4])

    

avg_show_comments = total_show_comments / len(show_posts)

avg_show_comments
# Calculate the amount of ask posts created during each hour of day and the number of comments received.

result_list = []



for item in ask_posts:

    result_list.append([item[6], int(item[4])])

    

counts_by_hour = {}

comments_by_hour = {}



for element in result_list:

    hour = datetime.strptime(element[0], "%m/%d/%Y %H:%M").strftime("%H")

    comment = element[1]

    

    if hour in counts_by_hour:

        counts_by_hour[hour] = counts_by_hour[hour] + 1

        comments_by_hour[hour] = comments_by_hour[hour] + comment

    else:

        counts_by_hour[hour] = 1

        comments_by_hour[hour] = comment

        

comments_by_hour
# Calculate the average amount of comments `Ask HN` posts created at each hour of the day receive.

avg_by_hour = []



for hour in counts_by_hour:

    avg_by_hour.append([hour, comments_by_hour[hour] / counts_by_hour[hour]])

    

avg_by_hour
swap_avg_by_hour = []



for item in avg_by_hour:

    swap_avg_by_hour.append([item[1], item[0]])

    

sorted_swap = sorted(swap_avg_by_hour, reverse = True)



sorted_swap
# Sort the values and print the the 5 hours with the highest average comments.



print("Top 5 Hours for Ask Posts Comments")

for index, value in sorted_swap[:5]:

    print("{}: {:.2f} average comments per psot".format(datetime.strptime(value, "%H").strftime("%H:%M"), index))    
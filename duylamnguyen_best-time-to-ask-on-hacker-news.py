from csv import reader



opened_file = open('../input/HN_posts_year_to_Sep_26_2016.csv')

read_file = reader(opened_file)

hn = list(read_file)

hn_header = hn[0]

hn_body = hn[1:]



print(hn_header)

print('\n')

print(hn_body[:5])
ask_hn = []

show_hn = []

other_hn = []



for row in hn_body:

    title = row[1].lower()

    if title.startswith('ask hn'):

        ask_hn.append(row)

    elif title.startswith('show hn'):

        show_hn.append(row)

    else:

        other_hn.append(row)



print('Number of Ask HN posts: ', len(ask_hn))

print('Number of Show HN posts: ', len(show_hn))

print('Number of Other HN posts: ', len(other_hn))
total_ask_comment = 0

total_show_comment = 0



for row in ask_hn:

    num_comments = int(row[4])

    total_ask_comment += num_comments

    

for row in show_hn:

    num_comments = int(row[4])

    total_show_comment += num_comments

    

avg_ask_comment = total_ask_comment / len(ask_hn)

avg_show_comment = total_show_comment / len(show_hn)



print('Average number of comment in Ask HN: ', avg_ask_comment)

print('Average number of comment in Show HN: ', avg_show_comment)
import datetime as dt



comment_by_hour = {}

count_by_hour = {}



for row in ask_hn:

    created_at = row[6]

    num_comments = int(row[4])

    hour = dt.datetime.strptime(created_at, '%m/%d/%Y %H:%M').strftime('%H')

    if hour not in count_by_hour:

        comment_by_hour[hour] = num_comments

        count_by_hour[hour] = 1

    else:

        comment_by_hour[hour] += num_comments

        count_by_hour[hour] += 1



print('Total number of comments by hour:\n', comment_by_hour)
avg_comment_hour = []



for key in comment_by_hour:

    avg_comment_hour.append([key, comment_by_hour[key] / count_by_hour[key]])

    

avg_display = []



for element in avg_comment_hour:

    hour = element[0]

    avg = element[1]

    avg_display.append([avg, hour])

    

avg_display = sorted(avg_display, reverse = True)



print('Top 5 Average Number of Comments by Hour:\n')

for element in avg_display[:5]:

    print('{}:00 : {:.2f} average comments per post'.format(element[1],element[0]))
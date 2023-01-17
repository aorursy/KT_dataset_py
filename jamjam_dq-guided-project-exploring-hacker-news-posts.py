# Read data
from csv import reader
opened = open('../input/HN_posts_year_to_Sep_26_2016.csv')
file = list(reader(opened))
hn = file[1:]
headers = file[0]

print(headers)
print(hn[:5])
ask_posts =[]
show_posts = []
other_posts = []

for row in hn:
    title = row[1]
    if title.lower().startswith('ask hn'): ask_posts.append(row)
    elif title.lower().startswith('show hn'): show_posts.append(row)
    else: other_posts.append(row)
        
print('There are ',len(ask_posts), 'in ask_posts.')
print('There are ',len(show_posts), 'in show_posts.')
print('There are ',len(other_posts), 'in other_posts.')
# find the average number of comments in Ask hn posts
total_ask_comments = 0

for row in ask_posts:
    num_comments = int(row[4])
    total_ask_comments += num_comments

avg_ask_comments = total_ask_comments/len(ask_posts)
# find the average number of comments in Show hn posts
total_show_comments = 0

for row in show_posts:
    num_comments = int(row[4])
    total_show_comments += num_comments

avg_show_comments = total_show_comments/len(show_posts)
print('The average comments in ask HN posts is ', round(avg_ask_comments,2), '.')
print('The average comments in show HN posts is ', round(avg_show_comments,2), '.')
import datetime as dt

# extract time created and the number of comments in ask HN posts
result_list = []

for row in ask_posts:
    created_at = row[-1]
    num_comment = int(row[4])
    result_list.append([created_at, num_comment])
# create frequency table by hour with total comments
counts_by_hours = {}
comments_by_hours = {}

for item in result_list:
    hour = dt.datetime.strptime(item[0], '%m/%d/%Y %H:%M').hour
    if hour in counts_by_hours: 
        counts_by_hours[hour] += 1
        comments_by_hours[hour] += item[1]
    else: 
        counts_by_hours[hour] = 1
        comments_by_hours[hour] = item[1]
# find the average number of comments by hour
avg_by_hour = []
for item in counts_by_hours:
    avg_by_hour.append([item, float(comments_by_hours[item])/float(counts_by_hours[item])])
# swap columns
swap_avg_by_hour=[]
for item in avg_by_hour:
    swap_avg_by_hour.append([item[1],item[0]])
sorted_swap = sorted(swap_avg_by_hour, reverse=True)
print('Top 5 Hours for Ask Post Comments:', '\n')
for i in range(5):
    print('{hour}:00: {num: .2f} average comments per post.'.format(hour=sorted_swap[i][1], 
                                                               num=sorted_swap[i][0]))
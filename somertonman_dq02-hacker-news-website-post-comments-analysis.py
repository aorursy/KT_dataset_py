import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from csv import reader

file='/kaggle/input/sample-hacker-news/hacker_news.csv'

hn=list(reader(open(file)))[1:]

headers=list(reader(open(file)))[0]

#hn[:5]
# Headers column

headers
ask_posts=[]

show_posts=[]

other_posts=[]



for i in hn:

    title=i[1].lower()

    if title.startswith('ask hn'):

        ask_posts.append(i)

    elif title.startswith('show hn'):

        show_posts.append(i)

    else:

        other_posts.append(i)

print ('Ask HN posts count: \t',len(ask_posts))

print ('Show HN posts count: \t',len(show_posts))

print ('Other posts count: \t',len(other_posts))

print("_"*30)

print ('Total posts count: \t',len(hn))
total_number_of_comments=0

for i in hn:

    comments_for_a_post =int(i[4])

    total_number_of_comments+=comments_for_a_post

    



def comments_counter(list):

    comments_total_for_category=0

    for i in list:

        comments_for_a_post =int(i[4])

        comments_total_for_category+=comments_for_a_post

    comments_percentage=round(comments_total_for_category/total_number_of_comments*100)

    avg_for_a_post=round(comments_total_for_category/len(list))

    return comments_total_for_category,comments_percentage,avg_for_a_post



ask_cnt=comments_counter(ask_posts)

show_cnt=comments_counter(show_posts)

other_cnt=comments_counter(other_posts)
print('Comments for Ask posts: {total}({percent}%), with average {post} for a post.'.format(post=ask_cnt[2],total=ask_cnt[0],percent=ask_cnt[1]))

print('Comments for Show posts: {total}({percent}%), with average {post} for a post.'.format(post=show_cnt[2],total=show_cnt[0],percent=show_cnt[1]))

print('Comments for Other posts: {total}({percent}%), with average {post} for a post.'.format(post=other_cnt[2],total=other_cnt[0],percent=other_cnt[1]))

import datetime as dt

z=ask_posts[5][6]

dt_template="%m/%d/%Y %H:%M"
dt.datetime.strptime(z,dt_template)
dt_template="%m/%d/%Y %H:%M"



n_comments_by_hour=[]



for i in ask_posts:

    date_row=i[6]

    date_conv=dt.datetime.strptime(date_row,dt_template)

    hour=date_conv.hour

    n_comments=int(i[4])

    n_comments_by_hour.append((hour,n_comments))

    

    

n_comments_by_hour[:7]
comments_by_hour={}

counts_by_hour={}



for i in n_comments_by_hour:

    hour=i[0]

    comments=i[1]

    if hour not in counts_by_hour:

        counts_by_hour[hour]=1

        comments_by_hour[hour]=comments

    else:

        counts_by_hour[hour]+=1

        comments_by_hour[hour]+=comments

avg_by_hour={}



for i in range(24):

    avg_by_hour[i]=comments_by_hour[i]/counts_by_hour[i]



import operator

sorted_avg_by_hour = sorted(avg_by_hour.items(), key=operator.itemgetter(1),reverse=True)





print ("Hours to publish a post with most number of comments (top five):")

for i in sorted_avg_by_hour[:5]:

    print ("Hour: {hour}:00, number of comments on average: {comments:.2f}".format(hour=i[0],comments=i[1]))

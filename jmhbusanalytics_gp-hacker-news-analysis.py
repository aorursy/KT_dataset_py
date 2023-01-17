#! pip install datetime
#import the needed modules

from csv import reader

import datetime as dt



# import the data sent into list

opened_file = open('../input/hacker-news/hacker_news.csv')

read_file = reader(opened_file)

hn = list(read_file)



#check the first 5 rows

hn[:5]
# put header in variable

header = hn[0]

print(header)

print('\n')

print(' the number of rows in the dataset are ' + str(len(hn)))





#remove header from the dataset and check its length

hn = hn[1:]

print(hn[0])

print('\n')

print(' the number of rows in the dataset are ' + str(len(hn)))

#Seperate posts based on Ask hn, show hn or other/ find their row length

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

        

        

print(ask_posts[:2])

print(' The number of "ask" posts are ' + str(len(ask_posts)))

print('\n')  

print(show_posts[:2])

print( 'The number of "show" posts are '  + str(len(show_posts)))

print('\n')

print(other_posts[:2])

print( 'The number of "other" posts are '  + str(len(other_posts)))
#determine if ask posts or show posts recieve more comments on average

#average ask comments

total_ask_comments = 0



for row in ask_posts:

    total_ask_comments += int(row[4])



avg_ask_comments = total_ask_comments/ len(ask_posts)  



print('The total of "ask" comments are ' +  str(total_ask_comments))

print('The average number of "ask" comments are ' + str(round(avg_ask_comments,1)))



#average show comments

total_show_comments = 0



for row in show_posts:

    total_show_comments += int(row[4])

    

avg_show_comments = total_show_comments / len(show_posts)



print('\n')

print('The total of "show" comments are ' +  str(total_show_comments))

print('The average number of "show" comments are ' + str(round(avg_show_comments,1)))





    

    
#calculating the amount of ask posts and comments created at each hour of the day

# Create a list of list with the date/time the post was sumbitted & the number of comments

result_list = []



for row in ask_posts:

    created_at = row[6]

    num_comments = int(row[4])

    result_list.append([created_at,num_comments])

   

    

print(result_list[:10])
#Create dictionaries containing number of post and comments for each hour of the day

counts_by_hour = {}

comments_by_hour = {}

date_format = "%m/%d/%Y %H:%M"



for row in result_list:

    date = row[0]

    comment = row[1]

    time = dt.datetime.strptime(date, date_format).strftime('%H')

    if time not in counts_by_hour:

        counts_by_hour[time] = 1

        comments_by_hour[time] = comment

    else:

        counts_by_hour[time] += 1

        comments_by_hour[time] += comment

print('The number of "ask" posts created for each hour of the day' + '\n') 

print(counts_by_hour)

print('\n')

print('The number of "ask" comments for each hour of the day' + '\n')

print(comments_by_hour)

    

    

    
#calculate the average number of comments per post for post created during each hour of the day

avg_by_hour = []



for each_hour in comments_by_hour:

    avg_by_hour.append([each_hour, round(comments_by_hour[each_hour] / counts_by_hour[each_hour],1)])



    

print('The average number of "ask" comments per a post in each hour of the day' + '\n')

print(avg_by_hour)
#sort the average comments in each hour of the day from highest to lowest (descending order)

swap_avg_by_hour = []



for row in avg_by_hour:

    swap_avg_by_hour.append([row[1], row[0]])



print(swap_avg_by_hour)

print('\n')



sorted_swap = sorted(swap_avg_by_hour, reverse = True)

print(sorted_swap)

print('\n')



#find top 5 highest avg of comments for an hour within a day

print('Top 5 Hours for Ask Posts Comments' + '\n')



for avg, hour in sorted_swap[:5]:

    avg_comments = avg

    each_hour = dt.datetime.strptime(hour,'%H').strftime('%H:%M')

    template = "{time} {comments:.2f} average comments per post"

    output = template.format(time = each_hour, comments = avg_comments)

    print(output)

    
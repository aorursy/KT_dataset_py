import csv as c

from csv import reader

import datetime as dt
#Read the hacker_news.csv file in as a list of lists.

open_file = open("../input/hacker-news-posts/HN_posts_year_to_Sep_26_2016.csv")

read_file = reader(open_file)

hn = list(read_file)

hn[0:5]
#Remove the first row from hn

hn = hn[1:]
#Check if the header is removed

hn[0:5]
#Create three empty lists called ask_posts, show_posts, and other_posts

ask_posts = []

show_posts = []

other_posts = []



#Assign the title in each row to a variable named title.

for row in hn:

    title = row[1]

    

#If the lowercase version of title starts with ask hn, append the row to ask_posts.

    if (((title.lower()).startswith("ask hn")) == True):

        ask_posts.append(row)

        

#Else if the lowercase version of title starts with show hn, append the row to show_posts.

    elif (((title.lower()).startswith("show hn")) == True):

        show_posts.append(row)

        

#Else append to other_posts.

    else:

        other_posts.append(row)

        

#Check the number of posts in ask_posts, show_posts, and other_posts

print(len(ask_posts))

print(len(show_posts))

print(len(other_posts))
print(len(ask_posts + show_posts+ other_posts))
#Find the total number of comments in ask posts 

total_ask_comments = 0



#Use a for loop to iterate over the ask posts

for row in ask_posts:

    num_comments = int(row[4])

    total_ask_comments += num_comments

    

print(total_ask_comments)
#Compute the average number of comments on ask posts

avg_ask_comments = total_ask_comments/len(ask_posts)

print(avg_ask_comments)
#Find the total number of comments in show posts and assign it to total_show_comments.

total_show_comments = 0



#Use a for loop to iterate over the show posts.

for row in show_posts:

    num_comments = int(row[4])

    total_show_comments += num_comments

    

print(total_show_comments)
#Compute the average number of comments on show posts

avg_show_comments = total_show_comments/len(show_posts)

print(avg_show_comments)
#Create an empty list and assign it to result_list. This will be a list of lists.

#Iterate over ask_posts and append to result_list a list with two elements:

#The first element shall be the column created_at

#The second element shall be the number of comments of the post.

result_list = []

for i in ask_posts:

    created_at = i[6]

    num_comments = int(i[4])

    result_list.append((created_at,num_comments))

result_list
#Create two empty dictionaries called counts_by_hour and comments_by_hour

counts_by_hour = {}

comments_per_hour = {}



#Extract the hour from the date, which is the first element of the row.

#Use the datetime.strptime() method to parse the date and create a datetime object.

for row in result_list:

    h = row[0]



#Use the string we want to parse as the first argument and a string that specifies the format as the second argument.

#Use the datetime.strftime() method to select just the hour from the datetime object.



#If the hour isn't a key in counts_by_hour:

##Create the key in counts_by_hour and set it equal to 1.

##Create the key in comments_by_hour and set it equal to the comment number.

    date_str = dt.datetime.strptime(h,"%m/%d/%Y %H:%M")

    posts_created = date_str.strftime("%H")



    comments_created = row[1]

    

    if posts_created not in counts_by_hour:

        counts_by_hour[posts_created] = 1

        comments_per_hour[posts_created] = comments_created

        



#If the hour is already a key in counts_by_hour:

##Increment the value in counts_by_hour by 1.

##Increment the value in comments_by_hour by the comment number.

    else:

        counts_by_hour[posts_created] += 1

        comments_per_hour[posts_created] += comments_created
#calculate the average number of comments per post for posts created during each hour of the day.

counts_by_hour
comments_per_hour
#calculate the average number of comments per post for posts created during each hour of the day

avg_by_hour = []

for i in counts_by_hour:

    average = comments_per_hour[i]/counts_by_hour[i]

    avg_by_hour.append([i,average])

avg_by_hour
#Create a list that equals avg_by_hour with swapped columns

swap_avg_by_hour = []

for row in avg_by_hour:

    swap_avg_by_hour.append([row[1],row[0]])

swap_avg_by_hour
#Use the sorted() function to sort swap_avg_by_hour in descending order

sorted_swap = sorted(swap_avg_by_hour, reverse=True)
print("Top 5 Hours for Ask Posts Comments")
#Use the str.format() method to print the hour and average

#use the datetime.strptime() constructor to return a datetime object 

#use the strftime() method to specify the format of the time.

#use {:.2f} to indicate that just two decimal places should be used

for each in sorted_swap[0:5]:

    hour = dt.datetime.strptime(each[1],"%H")

    hour = hour.strftime("%H:%M")

    a = str.format("{hour}: {comments:.2f} average comments per hour", hour = hour, comments = (each[0]))

    print(a)
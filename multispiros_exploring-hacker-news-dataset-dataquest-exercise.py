from csv import reader as read

hn = list(read(open("../input/hacker-news-posts/HN_posts_year_to_Sep_26_2016.csv")))
for row in hn[0:5]:

    print(row,"\n")
headers = hn[0]

hn=hn[1:]

print(headers,"\n")

for row in hn[:5]:

    print(row,"\n")
ask_posts = []

show_posts = []

other_posts = []

for row in hn:

    title = row[1]

    if (((title.lower()).startswith("ask hn")) == True):

        ask_posts.append(row)

    elif(((title.lower()).startswith("show hn")) == True):

        show_posts.append(row)

    else:

        other_posts.append(row)
print("total posts: ",len(hn))
print("ask posts:",len(ask_posts),"- show posts:",len(show_posts),"- other posts:",len(other_posts))
#first we will find total comments for ask

total_ask_comments = 0

for row in ask_posts:

     total_ask_comments+=int(row[4])

#then we will find average

avg_ask_comments = total_ask_comments/len(ask_posts)

print("Average comments for ask:",avg_ask_comments)
#Now we will find total comments for show

total_show_comments=0

for row in show_posts:

     total_show_comments+=int(row[4])

#then we will find average

avg_show_comments = total_show_comments/len(show_posts)

print("Average comments for show:",avg_show_comments)
import datetime as dt

result_list=[]



for row in ask_posts:

    temp=[]

    temp.append(row[6])

    temp.append(int(row[4]))

    result_list.append(temp)

    

counts_by_hour = {}

comments_by_hour={}
for row in result_list:

    temp = dt.datetime.strptime(row[0],"%m/%d/%Y %H:%M")

    temp = temp.strftime("%H")

    if temp not in counts_by_hour:

        counts_by_hour[temp]=1

        comments_by_hour[temp] = row[1]

    else:

        counts_by_hour[temp]+=1

        comments_by_hour[temp]+= row[1]

    
avg_by_hour = []

for row in counts_by_hour:

    avg_by_hour.append([row,comments_by_hour[row]/counts_by_hour[row]])
for row in avg_by_hour:

    print(row)
swap_avg_by_hour =[]

for row in avg_by_hour:

    swap_avg_by_hour.append([row[1],row[0]])

for row in swap_avg_by_hour:

    print(row)
sorted_swap = sorted(swap_avg_by_hour,reverse=True)
print("Top 5 Hours for Ask Posts Comments")
for row in sorted_swap[:5]:

    temp  = dt.datetime.strptime(row[1],"%H")

    temp = temp.strftime("%H:%M")

    print("{a} : {b:.2f} average comments per post ".format(a=temp,b=row[0]))
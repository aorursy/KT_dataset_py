from csv import reader

opened_file = open('../input/hacker-news-posts/HN_posts_year_to_Sep_26_2016.csv', encoding = "utf8")

read_file = reader(opened_file)

hn = list(read_file)

header = hn[0:1]

hn = hn[1:]
def explore_data(a_list, start =0 , end = 10, rows_and_columns = False):

    sliced_data = a_list[start:end]

    for x in sliced_data:

        print(x)

        print('\n')

    if rows_and_columns:

        print('Number of rows are - ', len(a_list))

        print('\n')

        print('Number of columns are - ', len(a_list[0]))
print(header,'\n')

explore_data(hn, 0, 5, rows_and_columns = True)
ask_posts = []

show_posts = []

other_posts = []



for x in hn:

    title = str(x[1])

    if (title.lower()).startswith('ask hn'):

        ask_posts.append(x)

    elif (title.lower()).startswith('show hn'):

        show_posts.append(x)

    else:

        other_posts.append(x)



print(len(ask_posts),'\n')

print(len(show_posts),'\n')

print(len(other_posts),'\n')
def find_average(a_list, index):

    total = 0

    for x in a_list:

        if x[index] != '':

            temp = int(x[index])

            total += temp

    return total/len(a_list)

    



avg_ask_comments = find_average(ask_posts,4)

avg_show_comments = find_average(show_posts,4)



print('Average Ask Comments are :', avg_ask_comments, '\n')

print('Average Show Comments are :',avg_show_comments, '\n')
import datetime as dt



result_list = [] #Taking only Created at and number of comments

for x in ask_posts:

    temp1 = x[6]

    temp2 = int(x[4])

    result_list.append([temp1, temp2])

    

counts_by_hour = {}

comments_by_hour = {}



for x in result_list:

    time = x[0]

    dt_time = dt.datetime.strptime(time,"%m/%d/%Y %H:%M")

    temp = dt_time.strftime("%H")

    if temp in counts_by_hour:

        counts_by_hour[temp] += 1

        comments_by_hour[temp] += x[1]

    else:

        counts_by_hour[temp] = 1

        comments_by_hour[temp] = x[1]



avg_by_hour = {}

for x in counts_by_hour:

    avg_by_hour[x] = comments_by_hour[x]/counts_by_hour[x]



def sort_dict(a_dict, descending = True):

    temp_list = []

    for x in a_dict:

        tuple_a = a_dict[x], x

        temp_list.append(tuple_a)

    if descending:

        sorted_list = sorted(temp_list, reverse = True)

    else:

        sorted_list = sorted(temp_list)

    return list(sorted_list)



avg_by_hour_list = sort_dict(avg_by_hour)

counter = 1

for x in avg_by_hour_list:

    if counter <= 5:

        time = dt.time(hour = int(x[1])).strftime("%H:%M")

        print("{counter}. {time} had average comments per post : {avg:.2f}".format(time = time, avg = x[0], counter = counter) )

        counter += 1
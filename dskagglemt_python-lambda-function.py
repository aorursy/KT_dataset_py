#multiply

multiply = lambda x,y: x*y

multiply(205,311)
# power of

power = lambda x,y : x*y

power(2,3)
some_num = [2,4,6,8]

double = map(lambda x: x+x, some_num)

print(double)   # This will return object details

print(list(double))  # to print actual output we have to use list. 
strings = ["My", "python", 'is', 'GreaT']

cap = map(lambda x: str.upper(x), strings)

print(cap)

print(list(cap))
attendance = [35,39,32,37,30,33]

print(sorted(attendance)) # Original data-set is not impacted.

print(attendance)
print(sorted(attendance, reverse = False)) # Original data-set is not impacted.

print(attendance)
print(sorted(attendance, reverse = True)) # Original data-set is not impacted.

print(attendance)
attendance.sort() # Original data-set is impacted.

print(attendance)
attendance.sort(reverse = True) # Original data-set is impacted.

print(attendance)
attendance = [35,39,32,37,30,33]

attendance.sort(key = lambda x: x*1.5)

attendance
class_attendance = [('9A', 35),('9B', 37), ('9C',30), ('9D',32),('9E',34)]

class_attendance
sorted(class_attendance, key = lambda x: x[1])
attendance = [35,39,32,37,30,33]

above_35 = filter(lambda x: x >=35, attendance)

list(above_35)
countries = ["India", "US", "UK", "France", "China", "Germany", "UAE"]

count_grt3 = list(filter(lambda x: len(x) > 3, countries))

count_grt3
from functools import reduce
nums = [10,20,22,25,29,35]

sum_all = reduce(lambda x,y: x+y, nums)

sum_all

# Here the results of previous two elements are added to the next element and this goes on till the end of the list like (((((10+20)+22)+25)+29)+35).
max_value = reduce(lambda x,y: max(x,y), nums)

min_value = reduce(lambda x,y: min(x,y), nums)
print(max_value, min_value)
# lambda <arguments> : <Return Value if condition is True> if <condition> else <Return Value if condition is False>

nums = [10,20,55,25,29,35]

max_value = reduce(lambda x,y: x if x > y else y , nums)

max_value
scores = [[1,35,80], [2,32,75], [3,30,82],[4,33,75], [5,37,60]]

# Lets assume above is a data of a Student with his ID, Attendance, and Marks in exam.

# Here we have to give 2 additional marks if Attendance is more than 35, and if less then reduce marks by 2.



avg = 35

newmarks = map(lambda x: x[2]+2 if x[1] >= avg else x[2]-2, scores)

list(newmarks)
sales = [

            {'country': 'India', 'sale' : 150.5},

            {'country': 'Chine', 'sale' : 200.2},

            {'country': 'US', 'sale' : 300.3},

            {'country': 'UK', 'sale' : 400.6},

            {'country': 'Germany', 'sale' : 500.9}

        ]

sales
# List out the Countries

country_key = list(map(lambda x: x['country'], sales))

country_key
# List out the sales

country_sales = list(map(lambda x: x['sale'], sales))

country_sales
India_sales = list(filter(lambda x: x['country'] == 'India', sales))

India_sales
# Get only high sales

high_sales = list(filter(lambda x: x['sale'] > 250, sales))

high_sales
list1 = [20,25,30,35,40]

list2 = [50,55,60,65,70]
# add tow list

add_list = list(map(lambda x,y: x+y, list1, list2))

add_list
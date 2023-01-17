%run '~/code/py/life.py'
%history # Gives history of commands run in notebook
s1 = 'Vaibhav'
print (s1[4])   # prints char at 4th index
print (s1[:4])  # prints chars before 4th index, 4th index excluded.

# Same for -ve index
print (s1[-2])   # prints char at 2nd last index
print (s1[:-2])  # prints chars before 2nd last index, 2nd last index excluded.

print ( len(s1) ) # prints length, number of characters in string.

str.upper(s1)    # same as below 
s1.upper()       # same as above

strList = list(s1) # converts to list data type, array of each char in string
print (strList)

# Formatting
myFormat = '%.2f %% this as fraction, and string %s and value INR%d'
myFormat %(2,1,4.8) # takes 3 args, convers and prints as per definition.


# find in string
'bh' in s1
type(float('23'))
my_list = [1,2,3,'hi',[2,3,4]]
# mix data type list
# list is collection of objects
print (my_list)

my_dict = {'k1':'v1' , 'k2':'v2'}
# Key value pair
print (my_dict)


my_tuple = (1,2,3)
# tuples are fixed values and cannot be assigned
print (my_tuple)

my_set = {1,2,3,1,2,1,2,1,2,1,2,1,3}
# set can have only unique values
# for above result is only 1,2,3
print (my_set)
a_list = ['amit', 'pranav', 'ramesh', 123, 22.4]
a_list.append(5)
print (a_list)

a_list.reverse() # reverses in place, returns nothing
print (a_list)

# a_list.sort()
print (a_list)

sorted('hello World 4')
sqr = [x**2 for x in range(5)] # This gives list with squares of numbers 0 to 4
print (sqr)
# With Condition
input_list = [1,2,3,4,5,6,7,8,9,8,7,56,4,32]
even = [var for var in input_list if var % 2 == 0]
print (even)
# list comp dict

state = ['Gujarat', 'Maharashtra', 'Rajasthan'] 
capital = ['Gandhinagar', 'Mumbai', 'Jaipur'] 
  
dict_using_comp = {key:value for (key, value) in zip(state, capital)} 
  
print(dict_using_comp) 
def my_sqr(num):
  return num**2

seq = [1,2,3,4,5]

map(my_sqr,seq) # It computes my_sqr for all items in seq.
lambda num: num*2
map(lambda num: num*2, range(1,5)) # square for list
map(lambda x, y: x + y,[1,20],[4,50])
filter(lambda num: num%2 == 0,seq)
my_dict = {
    'k1': 'value1',
    'k2': 'value2',
    'k3': 'value3'
}
my_dict.items() # returns list of tuples

print (type(my_dict))               # Dict is list of tuples of strings
print (my_dict.items())             # list of tuples of strings
print (type(my_dict.items()))
# print (type(my_dict.items()[0]))    # tuple of strings
# print (type(my_dict.items()[0][1])) # String
sorted(my_dict.values(), reverse=True)
def my_square(num):
  """
  This is DocString, shows up in help using ? or shift tab.
  Can be multiline
  This func squares a number
  """
  return num**2

my_square(4)
def my_sum(a=5, b=4, *args, **kwargs):
    print ('Arguments passed:', args)
    print ('Key With Args passed', kwargs)
    sum = a + b
    for n in args:
        sum += n
    return sum

my_sum(2, 4, 3, 1, key0='val0', key1='val1')
bool(1.2) #returns boolean or not
x=2
x**4 # to the power of
True and not False # keywords and not
type(chr(97) ) # keywork chr
sum(range(0,10)) # last, that is 10th index, is not included
%echo "hello world" > new.txt
handle = open('./new.txt', 'r')

for line in handle:
    print (line)
handle.close()
type(handle)
handle.name
handle.mode
new_file = open('new.txt', 'w')
new_file.write('some text')
new_file.close()
%cat new.txt
def average( numList ):
    # Raises TypeError or ZeroDivisionError exceptions.
    sum= 0
    for v in numList:
        sum = sum + v
    return float(sum)/len(numList)

def averageReport( numList ):
    try:
        print ("Start averageReport")
        m = average(numList)
        print ("Average = ", m)
    except (TypeError, ex):
        print ("TypeError: ", ex)
    except (ZeroDivisionError, ex):
        print ("ZeroDivisionError: ", ex)
    finally:
         print ("Finish block always runs")

list1 = [10,20,30,40]
list2 = []
list3 = [10,20,30,'abc']

averageReport(list1)
state = ['Gujarat', 'Maharashtra', 'Rajasthan'] 
capital = ['Gandhinagar', 'Mumbai', 'Jaipur'] 
  
output_dict = {} 
  
# Using loop for constructing output dictionary 
zip(state, capital)
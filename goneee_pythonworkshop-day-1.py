# Assign 3 to the variable a

a=3 # Run the cell, nothing will appear as output
# The Kaggle notebook will print the last expression (variable) in each cell

# output the result of without using print

a
# Data Type of 'a'

type(a)
# Store the product of 4 and 5 in a

a = 4 * 5

a
# Just one of many possible solutions

five_to_power_of_8 = 5 ** 8

five_to_power_of_8
# python identifies a block of code using the indentation

# indentation should be consistent

# while there are many preferences, one recommendation is to indent by 4 spaces



print("Wow") # print wow 

if 1 == 3:   # opening if-statement testing if 1 is equal to 3. NB. The COLON is important

    print("1 == 3") # if true print "1==3"

else:        # else

    print(" 1 is not equal to 3") #( inside the else block)

    print("Sure") # inside the else block

print("Yup") # this is not inside the else block
name=input("What is your name?")

name
# One of many possible solutions

# but can you fix my output, it doesn't exactly match the example in the question

number = int(input("Give me a number: "))

print(number * 3)

if number &1:

    print("-"*10)

    print("---"+"Odd way to start"+"---")

    print("-"*10)

else:

    print("-"*10)

    print("---"+"Yea Even"+"---")

    print("-"*10)
# Exampl

num = 4

if num > 5 and num > 0: #NB the use of "and" and not "&&"

    print("%d is really greater" % num)

else:

    print("Trying some other logic")

    if num < 4: #NB that this if is inside the indentation block of the else block of the first if

        print("This number is less than 4")

    else:

        print("{0} is greater than or equal to 4".format(num))

        
"Yes" if 5 > 0 else "No" # ternary ....equivalent to " 5 > 0 ? "Yes" : "No""
# Typical foreach loop

for i in [0,1,2,3,4]: # for(int i=0;i<5;i++) <-- we don't have this anymore

    print(i)
# Typical foreach loop

for i in range(0,5): # for(int i=0;i<5;i++) <-- we use the range function to generate the above sequence

    print(i)
# Example of a while-loop, what is it doing?

num =0

while num < 7:

    num = num +1

    print(num)
x = [-2,-1,0,1,2,3] # create and initialize a list

y=[] #create and initialize an empty list

x[3]=5 # assin 5 to position 3 within the list

x.append(7) # append 7 to the end of the list - find more list operations with help(list)

x
# One of many possible solutions

# The question is what is each part of each line doing

x, y=[], [] #tuple unpacking

for num in range(-5,8):

    num =float(num)

    x.append(num)

    y.append(num**2)

print("X = %s \n Y = %s" % (x,y))
# Could it have been done in less lines #idk

x,y = [float(num) for num in range(-5,8)],[num**2 for num in range(-5,8)]; print("X = %s \n Y = %s" % (x,y))
# list comprehension

# Create another list from another list/iterator

# The "[]" are important

# Here we take the values generatd from range(3,8)

# And for each of them, call/store each value in num

# return the value stored in num -3

[num-3 for num in range(3,8) ]
range(-5,8)
# Shorthand contribution, what is it doing?

[*range(-5,8)]
# Finally a plot, 

import pandas as pd

pd.DataFrame({

    "x_column":x,

    "y_column":y

}).plot(kind="scatter",x="x_column",y="y_column",title="Always great to smile!")
# Create an intiialize a set

# Run "help(set)" for list of operations

set_a = set(['a','b','c','a'])

set_a
help(set_a.pop)
# Create and initialize a dictionary in python

my_phone={

    "model":"Huawei", # first key is "model" with a value of "Huawei"

    "space":6.4*10**6, #values may be numerical

    "operating_system":{ # the value for operating_system is another dictionary

        "name":"Android",

        "version":"oreo",

        "cpu":"ARMv7"

    },

    "missed_calls_from":[8763456746,8764234245] # what is the data type of the value in missed_calls_from

}

my_phone
#create a copy of the original dictionary

my_phone_2 = my_phone.copy()

my_phone_2
id(my_phone) != id(my_phone_2) # different objects in memory
print(my_phone["model"]) # access and print the model name

# Changed model to IPhone X Plus

my_phone["model"]="IPhone X Plus" # changed the model name to "IPhone X Plus"

# Changed CPU of operating system  to Snapdragon

# NB that 'cpu' is inside 'operating_system' and 'operating_system' is a key inside my_phone

my_phone['operating_system']['cpu']='Snapdragon'

my_phone
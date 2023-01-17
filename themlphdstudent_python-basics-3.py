a = 2



# check whether number equals to 2 or not

if a == 2:

  print(a)
num = 10



# equality checking

if num == 10:

  print('Number is equal to 10, is it true or false?', num == 10)

  

# output

# Number is equal to 10, is it true or false?



# inequality checking

if num != 11:

  print('Number is not equal to 10, is it true or false? ', num != 11)

  

# output

# Number is not equal to 10, is it true or false?  True



# multiple conditions

age = 18



# 'and' operator will check both conditions, if both conditions are true then only it run inner block.

if (age > 10) and (age < 20):

  print("Your age is between 10 to 20, is it true or false?", (age>10) and (age<20))

  

# output

# Your age is between 10 to 20, is it true or false? True



# another example of multiple condition with 'and' operator

if (age > 10) and (age < 15):

  print("Your age is between 10 to 15, is it true or false?", (age>10) and (age<15))

  

# output

# Our condition returns False so there is no output. 



# Let's try same condition with 'or' operator.

# While using 'or' operator if only one condition needs to be true for running code inside if statement.

if (age > 10) or (age < 15):

  print("Your age is between 10 to 15, is it true or false?", (age > 10) or (age < 15))

  

# output

# Your age is between 10 to 15, is it true or false? True



status = True



if status:

  print("Status is", status)



# output

# Status is True



# checking whether item is in the list or not



foods = ('pizza', 'falafel', 'carrot cake', 'cannoli', 'ice cream')



if 'pizza' in foods:

  print('I like pizza.')



# output

# I like pizza.



# even you can chech whether item is not in the list condition

if 'pizza' not in foods:

  print('I don\'t like pizza.')

  

# output

# here condition is false because 'pizza' is in the list
percentage = 77



if percentage < 35:

  print("You fail in exam.")

  

elif percentage >= 35 and percentage < 70:

  print("You pass in exam.")

  

else:

  print("You pass with first class.")

  

# output

# You pass with first class.


percentage = 67



if percentage < 35:

  print('You fail in exam.')

  

elif percentage == 35:

  print('You passed exam with passing percentage.')

  

elif percentage > 35 and percentage <= 50:

  print("Your percentage is between 35 to 50.")

  

elif percentage > 50 and percentage <= 70:

  print("Your percentage is between 51 to 70.")

  

elif percentage > 70 and percentage <= 100:

  print("Your percentage is between 71 to 100.")

  

# output

# Your percentage is between 51 to 70.
name = input("What\'s your name?")



print("Hello, ", name.title())


age = input("What is your age?")



print(type(age))

# output

# <class 'str'>



# convert age from string to int

age = int(age)



print(type(age))

# output

# <class 'int'>
count = 0



while count <= 10:

  print(count)

  count+=1



# output

# 1

# 2

# 3

# 4

# 5

# 6

# 7

# 8

# 9

# 10
message = "Where do you live?"

message+= "\n(Enter 'exit' to stop program)"



# this loop runs always

while True:

  # get country value from user

  country = input(message)

  

  # check whether user wrote exit

  if country == 'exit':

    #if user wrote exit break the loop

    break

    

  # if user does not wrote exit

  else:

    print('You live in ' + country.title() + ' !')



# output

# >>> Where do you live?

# (Enter 'exit' to stop program) -- Australia

# You live in Australia !

# >>> Where do you live?

# (Enter 'exit' to stop program) -- exit


number = 0



while number < 5:

  # increament number

  number += 1

  # check modulo of number

  # if modulo of number is equals to 0

  # then print message and continue

  if number % 2 == 0:

    print(str(number) + " is even number.")

    continue

  

# output

# 2 is even number.

# 4 is even number.
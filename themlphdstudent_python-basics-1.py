print("Hello World !!")
# This is comment.

print("You sucessful learn how to comment.")
a = 10
# integers

a = 1



# float

b = 1.1



# string

c = 'Hello World !'



# boolean

d = True
a = 'This is string'

b = "This is another string"



print(a)

print(b)

message = "HeLLo WoRLd !"



print("Title case : " + message.title())

print("Upper case : " + message.upper())

print("Lower case : " + message.lower())
first_name = "Durgesh"

last_name = "Samariya"



full_name = first_name + " " + last_name



print(full_name)
first_name = "Durgesh"

last_name = "Samariya"



# added tab space between first and last name

full_name1 = first_name + "\t"+ last_name



print(full_name1)



# added new line between first and last name

full_name2 = first_name + "\n" + last_name



print(full_name2)

s = ' This string contains white space. '



print(s.strip())

print(s.lstrip())

print(s.rstrip())
s = " Hello World !"

print("Length of string is ",len(s)) 
# integers

a = 2

b = 3



# add a and b

c = a + b

print("Addition is : " , c)



# subtract b from a

d = a - b

print("Subtraction is : " , d)



# multiplication 

e = a * b

print("Multiplication is : " , e)



# divide a from b

f = a / b

print("Division is : " , f)
# floats

a = 1.5

b = 0.5



# add a and b

c = a + b

print("Addition is : ", c)



# subtract b from a

d = a - b

print("Subtraction is : ", d)



# multiplication 

e = a * b

print("Multiplication is : ", e)



# divide a from b

f = a / b

print("Division is : ", f)

a = 2

b = 3



# following line returns error. Check in your console.

message = "Sum of " + a + " and " + b + " is " + (a+b) + "."

print(message)
# In python we need to covert each int or float when we are using them with string. To do that we have str() function.

message = "Sum of " + str(a) + " and " + str(b) + " is " + str(a+b) + "."

print(message)
# list that contains even numbers.

even = [2, 4, 6, 8, 10]

print(even)
# create list that contains one string, float and integer values.  

list = ['string', 1.25, 2]

print(list)

list = [1, 2, 3, 4, 5]



# print value at index 0

print(list[0])



# print value at index 2

print(list[2])



# print last item from the list

print(list[-1])



# similarly we can access second last item from list using -2, third item using -3 and so on

print(list[-2])
list = [2, 4, 6, 8, 10]

print(list)



# adding item in list



list.append(12)

print(list)



# inset item in list

# we are going to use insert(index, item) method.

list.insert(0, 0)

print(list)



# removing items from list



del list[0]

print(list)



# removing last element from the list using pop() method

list.pop()

print(list)



# removing item by value

list.remove(8)

print(list)
programming_languages = ['java', 'css', 'python', 'c', 'r']

print(programming_languages)



# sort in ascending order

programming_languages.sort()

print(programming_languages)



# sort in descending order

programming_languages.sort(reverse=True)

print(programming_languages)

list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print(len(list))
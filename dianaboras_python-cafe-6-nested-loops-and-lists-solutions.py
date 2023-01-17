# Your Code Goes Here

for i in range(0, 4):

    print("outer loop ", i)

    for j in range(1, 5):

        print(j)

        
# Your Code Goes Here

for i in range(0, 4):

    print("first loop ", i)

    for j in range(1, 5):

        print("second loop",j)

        for k in range(1,10):

            print("third loop", k)

        
# Your Code Goes Here

fruits = ["apple", "banana", 1]

print(fruits)
#Accessing the first element

fruits[0]
#Replacing an item

fruits[1] = "mango"

print(fruits)
####Multi-Dimensional List: 

l = [["list","of"], ["lists"]]
#Accessing elements

l[0][1]

l[1]
# Adding elements

a = [1,2,3,4,5]

b = [6,7,8,9,10]

c = [11,12,13,14,15]



#Single element

a.append(34)

print(a)



#Multiple elements

a.extend([6,7,8])

print(a)



#Concatenating lists

a+b+c
#Dropping elements



#Dropping by index

b = [10,11,12,13,14,15,15,15]



#Dropping the 4th element

b.pop(3)

print(b)

#obs: pop removes the last element if you don't specify the index value



#Dropping by value

b.remove(12)

print(b)
# Your Code Goes Here

basketball = ['Toronto', 'Raptors', 'NBA', 'Champions', 1]



for x in basketball:

    print(x)
#Nested lists



a = [[1,2,3],['a','b','c'],['cat','dog','mouse']]



#Accessing the first dimension of the list

for i in a:

 print(i)



#Accessing each element within each list of the multi-dimension list

for i in a:

    print(i)

    for j in i:

        print(j)
#Creating a list with common elements

aaa = [1,2,3,4]

bbb = [3,4,5,6,7,9]



#Create empty list

common_num = []



for a in aaa:

    for b in bbb:

        if (a==b):

            common_num.append(a)



print(common_num)
###Exercise 4 - Write a Python program separate even numbers from odd numbers



numbers = [0,1,5,7,88,90,34,55,67,890]



#Answer

odd_numbers=[]

even_numbers=[]



for i in range(0, len(numbers)):

    if (numbers[i] == 0):

        continue

    elif(numbers[i] % 2 != 0):

        odd_numbers.append(numbers[i])

    else:

        even_numbers.append(numbers[i])



print(odd_numbers)

print(even_numbers)
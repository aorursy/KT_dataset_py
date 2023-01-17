a=10

b=20

c=a+b

print(c)
a=10

a+=1

print(a)

print(3!=2)
first_name=input("What is your first name: ") #first_name

last_name=input("What is your last name: ") #last_name

print(f"My name is: {first_name} {last_name}")
num=input('enter a number: ')

#num+=10   error

num=int(num)

num+=10

print(num)
# Python always takes input as a string. Except string there also available integer, float, boolean types.



year=input("Enter your birth_year: ")

print(type(year)) # type() returns the type of the variable.
# Calculate your age.



year=input("Enter your birth_year: ")

print("Your cuurrent age is: {}".format(2019-int(year)))



'''

You can also use:

    float(): To convert into float.

    bool(): To convert into boolean.

'''
# ( Operator Precedence )

# x=2+(2-2)*2**2/2, calculate the value of x and print it. 



x=2+(2-2)*2**2/2

print(f"x = {x}")
# Input two numbers and swap them



x,y=input('Enter two numbers: ').split(',')

y,x=int(x),int(y)

print('x={} y={}'.format(x,y))
# Input a number and check whether it is odd or even and display accordingly.



a=int(input('enter a number: '))

if a%2==0:

    print('{} ia even'.format(a))

else:

    print('{} is odd'.format(a))
# Input your marks and give a review.



marks=float(input("Enter your marks: "))

if marks>0 and marks<50:

    print("BAD")

elif marks>50 and marks<80:

    print("Good")

elif marks>80 and marks<100:

    print("OUTSTANDING")

else:

    print("Invalid Marks !")
#Print the pattern: 



n=int(input('Enter value of n: '))

i=1

while i<=n:

    print('* '*i)

    i+=1
# Input a number and find the sum of its digits.



n=int(input('enter a no: '))

sum=0

while n>0:

    rem=n%10

    sum=sum+rem

    n=n//10

print('sum of digits is: ',sum)
# Input a number n and print all odd numbers upto n.



n=int(input('enter the range: '))

for i in range(1,n+1,2):

    print(i,end=' ')
#Print the pattern: 



n=int(input('Enter value of n: '))

for i in range (1,n+1):

    for j in range (i,n): # nested loop

        print(' ',end=' ')

    for k in range(1,i+1):

        print('{} '.format(k),end='')

    print()
# Input a number n and find its factorial using a user defined function int fact(int)



def fact(n):

    s=1

    for i in range(1,n+1):

        s=s*i

    return s

n=int(input('Enter a number: '))

print(f'The factorial of {n} is: {fact(n)}')
"""

Implement simple arithmetic calculator using user defined functions for each operation

(addition, subtraction, multiplication, division, modulus, exponent). You may use a dictionary to print the menu.

"""



def add(x, y):

    return x+y



def subtract(x, y):

    return x-y



def multiply(x, y):

    return x*y



def divide(x, y):

    return x/y



def mod(x,y):

    return x%y



def expo(x,y):

    return x**y



n1,n2 = input('Enter the two numbers  : ').split(',')

n1,n2=int(n1),int(n2)

while True:

    print("****MENU****\n1. Addition\n2. Subtraction\n3. Multiplication\n4. Division\n5. Modulus\n6. Exponent\n7. Exit\n")

    ch=int(input('enter your choice : '))

    if ch==1:

        print(add(n1,n2))

    elif ch==2:

        print(subtract(n1,n2))

    elif ch==3:

        print(multiply(n1,n2))

    elif ch==4:

        print(divide(n1,n2))

    elif ch==5:

        print(mod(n1,n2))

    elif ch==6:

        print(expo(n1,n2))

    elif ch==7:

        break

    else:

        print("Invalid input")
# Input a number n and find its factorial using recursion.



def fact(n):

    if n<1:

        return 1

    else:

        return n*fact(n-1)

n=int(input('Enter a number: '))

print(f'The factorial of {n} is: {fact(n)}')
# Input two strings and concatenate them.



str1=input('enter the first string:')

str2=input('enter the second string:')

str1=str1 + str2

print('the concatinated string: ',str1)
# Indexing

'''

Index:

    T y l e r  D u r d e n

    0,1,2...     ...-3,-2,-1

    

'''



name='Tyler Durden'

print(name[0])

print(name[-1])
# Slicing



name='Tyler Durden'

print(name)



name1=name[2:8]

print(name1)



name2=name[:-2]

print(name2)



name3=name[4:]

print(name3)



name4=name[:]

print(name4)
# Some string Methods:



name='Van Rossum'



print(len(name)) # to obtain length.
print(name.upper()) # to change into upper case.

print(name.lower()) # to change into lower case.

print(name.find('R'))
print(name.find('D')) # to find index of string.

print(name.replace('Rossum', 'Mukherjee')) # to replace a string.

#immutable

print(name)
#Create a list of individual characters from a string

str='abc'

print(list(str))
# Input a string and reverse it.



str=input('Enter a string: ')

s=''

for i in str:

    s=i+s

print("The reverse string is: ",s)
# Declaration of tuples.



t1 = (1,2,3)

t2 = (4,5,6)
# Tuple concatenation.



t3 = t1 + t2
#  Print.



print(t1)

print(t2)

print(t3)
# Tuple indexing.



print(t3[-3]) # It returns a new tuple
# Tuple slicing.



print(t3[1:4]) # It returns a new tuple
# 2D Tuples



t1=(1,2,3)

t2=(4,5,6)

t3=(t1,t2)

print(t3)
t3[0][0]
# Declaration of two dimentional list.



t=(

    (1,2,3),

    (4,5,6),

    (7,8,9)

)



print(t)



print(t[0][1])
# 2D-Tuple concatenation.



print(t + (10,11,12))
# 2D-Tuple indexing.



print(t[-2])

print(t[-2][1])
# 2D-Tuple slicing.



print(t[0:-1])
# List Declaration.



l = [1,2,3,4,5]



print(l)
# In list we can update a value of a list as it is immutable.



l = [1,2,3,4,5]

print(l)



# List Update.



l[1] = 20

print(l)
l1=[1,2,3]

l2=[4,5,6]

l3=[l1,l2]

print(l3)
print(l3[0][1])
# Declaration of two dimentional list.



l=[

    [1,2,3],

    [4,5,6],

    [7,8,9]

]



print(l)



print(l[0][1])
l = [1,2,3,4,5]

print(l)



# Append.

l.append(6)

print(l)



# We can also insert a number in any position using insert() method.

l.insert(0,100)

print(l)
l[::-1]
# Convert a tuple into a list.



t=(1,2,3,4,5)

print(list(t))



# Convert a list into a tuple.



l=[1,2,3,4,5]

print(tuple(l))
# Find maximum elements in a list.



numbers = [6,1,5,10,2]

max=numbers[0]

for number in numbers:

    if number>max:

        max=number

print(f" Maximum number in list is: {max}")
'''

    Find the sum of digits of a number using an user defined function digits(x) which returns a tuple containing the digits

    of the number.

'''

# Sumofdigits



def digits(x):

    t1=()

    while(x>0):

        t1=t1+(x%10,)

        x//=10

    return t1



n=int(input('Enter the number : '))



t2=digits(n)

print(t2)



sum=0

for i in t2:

    sum=sum+i

print(sum)
# Declaration



student={

    "name" : "Tyler Durden",

    "age" : 27,

    "marks" : 77.50,

    "email" : "tyler@gmail.com"

}
# Access value through a key



print(student["name"])
# " print(student["birth_date"]) " will give an error, because there is no key name " birthyear "



# Using get() method we can get a default value if the key is not pressent in the dictionary



print(student.get("birth_date","feb 2, 1998"))
# Update of a value



student["name"] = "Bob Biswas"

print(student["name"])
# Add a new key



student["birth_date"] = "aug 15, 1947"

print(student["birth_date"])
# Coverts a phone numbers into words



digits_to_words={

    "1" : "one",

    "2" : "two",

    "3" : "three",

    "4" : "four",

    "5" : "five",

    "6" : "six",

    "7" : "seven",

    "8" : "eight",

    "9" : "nine",

}



ph_number=input("Enter a phone number: ")

ph_words=""



for ch in ph_number:

    ph_words+=digits_to_words[ch] + " "

    

print("The phone number in words is: ",{ph_words})
# Declaration



s={1,2,3,4,5}

t={3,4,5,6,6,7}



#print



print(s)

print(t)
# Union



print(s.union(t))
# To add more than one value



t.update([8,9])



print(t)
# To remove a value



t.remove(6)



print(t)
# List coverstion to a set



s = [1,2,3,4,2,1,5]



s=set(s)



print(s)
name = ["Shahrukh", "Salman", "Amir", "Akshay"] 

rank = [3, 4, 1, 2 ] 

charge_per_film = [60, 50, 80, 50] 

  

obj=zip(name, rank, charge_per_film) 

  

# converting values to print as set 

obj=set(obj) 

  

print("The zipped result is : ",end="") 

print(obj)
n, r, ch = zip(*obj) 

print("The unzipped result: \n",end="") 

print(n) 

print(r) 

print(ch) 
name = ["Shahrukh", "Salman", "Amir", "Akshay"]

charge = [60, 50, 80, 50] 



for n,c in zip(name, charge): 

    print ("Actor : %s     Charge : %d" %(n, c)) 
def square(x): 

    return x*x; 

print(square(5)) 



  

f = lambda x: x*x 

print(f(7)) 
mylist = [5, 7, 22, 97, 54, 62, 77, 23, 73, 61] 

filtered_list = list(filter(lambda x: (x%2 != 0) , mylist)) 

print(filtered_list) 
def cube(n): 

    return n * n * n

  

list1 = [1, 2, 3, 4]

result = map(cube, list1) 

print(list(result))
t = (1, 2, 3, 4) 

result = map(lambda x: x * x * x, t) 

print(tuple(result)) 
list1 = [1, 2, 3] 

list2 = [4, 5, 6] 

  

result = map(lambda x, y: x + y, list1, list2) 

print(list(result)) 
'''# Create a class called Rectangle having two attributes – 

length and breadth and find the area and perimeter of the rectangle using two methods – showarea() and showperimeter()

--> Using Constructor'''

class rectangle:

    def __init__(self,l,b): #instance variable unique to each instance

        self.length = l

        self.breadth = b



    def showarea(self):

        return self.length*self.breadth

   

    def showperimeter(self):

        peri=2*(self.length + self.breadth)

        return peri



obj1=rectangle(10,5)

obj2=rectangle(12,6)

print(obj1.showarea())

print(obj1.showperimeter())

print(obj2.showarea())

print(obj2.showperimeter())
'''# Create a class called Rectangle having two attributes – 

length and breadth and find the area and perimeter of the rectangle using two methods – showarea() and showperimeter()

-->Not Using Constructor'''

class rectangle:

    length=10   #class variables shared by all instances

    breadth=5

    def showarea(self):

        return self.length*self.breadth

   

    def showperimeter(self):

        peri=2*(self.length + self.breadth)

        return peri



obj1=rectangle()

obj2=rectangle()

print(obj1.showarea())

print(obj1.showperimeter())

print(obj2.showarea())

print(obj2.showperimeter())
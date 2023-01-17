print("Hello World")
a = 10
b = 10.5

print(a)
print(b)
print(type(a))
print(type(b))
c = "hello_python"
print(type(c))
d = 5 + 3j
print(d)
print(type(d))
a = 100.5

print(a)
print(type(a))
name = raw_input("Enter the name of your organisation")
print("Hello",end=' ')
print(name)
name = input("Enter the name of your organisation")
print("Hello")
print(name)
a = float(input("Enter a Number"))
print(a)
print(a*a)

print(type(a))
print("The square of the number is " + str(a*a))
print("Hello","World","I","Love","Python",sep="+")
a = 10
b = 20
print(a,end=',')
print(b)
a = "Indians"
b = "Mangoes"

c = "Russians"
d = "Pizza"
print("{0} love {1}".format(c,d))
a = 10
b = 20

print("You entered %d and %d and %d"%(a,b,b))
a = 10
b = 21

print(a+b)
print(a-b)
print(a*b)
print("%0.4f"%(a/b))
print(a//b)
#Exponent/Power Function
print(a**b)
print(b%a)



a,b,c = 10,20,30.34
# This functions print the variables
print(a,b,c)

# This is a single comment !

"""This is a 
multiline 
comment """

myString = """This is 
a multiline string """

print(myString)


raining = True
temperature = 30


outing = raining and temperature <= 30
print(outing)


weather  = input()

if weather=="Rainy":
    print("Don't go outside")
    print("It is raining heavilly")
    
elif weather=="Cool":
    print("Lets play Cricket")

else:
    print("Lets go for shopping")



n = 10
i = 1

while i<=n:
    print("Step %d"%i)
    i = i + 1

print("Loop Finished")
    
    
for i in range(1,10,2):
    print(i,end=",")
no = 10
ans = 1

for i in range(1,11):
    ans *= i
    
print(ans)
for i in range(10,1,-2):
    print(i,end=',')
# 5 X 5 Matrix
n = 5
for x in range(n):
    for y in range(n):
        print(max(x+1,y+1,n-x,n-y),end=" ")
    print()

for i in range(1,10):
    if i==5:
        continue
    print(i)

print("Loop ends")
# Write a Program which prints all primes number upto N !
def helloFact():
    print("Hello Factorial")
    
helloFact()
    
    
def factorial(n):
    ans = 1
    for i in range(1,n+1):
        ans *= i
    return ans

print(factorial(5))


def isPrime(n):
    
    for i in range(2,n):
        if(n%i==0):
            return False
        
    return True


def printPrime(V):
    
    for i in range(1,V+1):
        if(isPrime(i)):
            print(i,end=',')
            
            
printPrime(100)
    
lang = "C++"
def say(x="Python"):
    print("I Love " +x)
    
say("JavaScript")
say()
print(lang)
def fact(n):
    #Base Case
    if(n==0):
        return 1
    #Rec Case
    return n*fact(n-1)

print(fact(6))
def myFunc(score,lang,rollNo):
    print("I scored %d in %s"%(score,lang))
    print(rollNo)
    
    
myFunc(rollNo=1010,lang="Python",score=10)
def fun(a,b,*x,**y):
    print(a)
    print(b)
    print(x)
    print(type(x))
    print(y)
    print(type(y))
    
    for k in y:
        print(k,y[k])
    
    
fun(1,2,3,4,10,14,shake="OreoShake",drink="lemonade",fruit="Mango")
a = "Coding Blocks"
print(a)
print(a[2])
print(a[-1])
print(len(a))
# Strings are Not Mutable!
a = "Code Blocks"
print(a)
for i in range(len(a)):
    print(a[i])
for c in a:
    print(c)
a = "Mango"
b = "Juice"
print(a + b)
print(a*5)
# Range Slicing 
a[1:-2]
a[ : ]
a[-2:]
a[1:3]
# Membership - Substring is present in bigger string
a = "Coding Blocks"
b = "Code"
if b not in a:
    print("Yes")
else:
    print("No")

    #'go' in a
# String Formating 
print("My favourite fruit is %s and it comes from %c"%("Mango","I"))
# Triple Quotes
para = """This is some
paragram written
here"""
print(para)
l = para.split()
print(l)
print(type(l))
print(type(l[0]))
para.splitlines()
fruit = "Mango"
fruit = fruit.upper()
fruit = fruit.lower()
print(fruit)
shake = "    Apple Shake   "
shake2 = shake
print(len(shake))
shake = shake.lstrip() #Removes the leading white space
shake = shake.rstrip() #Removes the ending white space
print(shake)
print(len(shake))
shake2 = shake2.strip() #Removes leading and trailing whites spaces 
print(shake2)
print(len(shake2))
a = "9318790"
a.isdigit()
a = "HNo"
a.isalpha()
a = "12abcA"
b = "    "
print(a.isalnum())
print(a.islower())
print(b.isspace())
a = "I love having Apple Juice, and I like eat green Apple"
print(a.find("Apple",30,len(a)))

print(a.index("Apple",30,len(a)))
# Difference is find() return -1, index throws an exception
a = a.replace("Apple","Mango")
print(a)
a.count("Mango")
l = a.split()
print(l)
b = "_"
b = b.join(l)
print(b)
name_org = "chiranjeev kumar"
name_org.title()
myList = [ 1,2,3.5,"Hello"]
print(myList)
print(type(myList))
l2 = list([1,2,3])
print(l2)
print(type(l2))
l3 = list(l2)
print(l3)

l3 = l3 + l2
print(l3)

l3.extend(l2)
print(l3)
# List of Square of the numbers from 1 to 5
l4 = [i*i for i in range(1,6)]
print(l4)
print(l4[0:3])
print(l4[-3:])
l = [1,2]
l.append(3)

l.append([1.0,2.1])
l += [4,5,6]
print(l)
l.insert(2,20)
print(l)
print(l[3][1])
del l[0]
print(l)
l.pop()
print(l)
l = ([1,2,3])
l = l*4
print(l)
l = ["Apple","mango","guava",80]
80 in l
for i in range(len(l)):
    print(l[i])
for x in l:
    print(x)
l = [4,3,2,16,18]
print(max(l))
print(min(l))
#Linear Search 
print(l.index(16))
l = sorted(l)
print(l)
l = [4,5,1,3,2]
l.sort(reverse=True)
print(l)
## Read a list of Numbers
numbers = [int(number)*int(number) for number in input().split()]
print(numbers)
print(type(numbers))
t = (1,2,3,"Hello")
print(t)
#tuples are immutable
#t[0] = 5
print(t[0])
# Convert a tuple into a list

l = list(t)
print(l)

l[0] = 5
print(l)
t2 = tuple(l)
print(t2)
#concatentation
print(t2 + t)

#Repetion
t2 = t2*3
print(t2)
t = (1,44,3,2)
print(min(t))
#for deleting the tuple
del t
print(min(t))
d = {"Mango":100,"Apple":80}
print(d)
#Look Up
print(d["Mango"])

d["Guava"] = 60
print(d)

d["Grape"] = [10,20]
print(d["Grape"])

d["Pineapple"] = {"Small":90,"Large":150}

print(d["Pineapple"]["Small"])
print(d.keys())
print(d.values())
print(type(d.get("Mangoes")))
if "Mangoes" in d:
    print("Price of Mango is %d"%(d["Mango"]))
else:
    print("Doesn't exist")
del d["Pineapple"]
l = list(d.items())
print(l)
d2 = {"Strawberry":95}

d.update(d2)
print(d)
print(len(d))
d.clear()
print(len(d))
l1 = ["Apple","Papaya","Guava","Banana"]
l2 = [100,120,30,50]

p = dict(zip(l1,l2))
print(p)
print(type(p))
for k in p.values():
    print(k)
s = set([15,1,2,3,4,3,1,1])
print(s)
if 14 in s:
    print("present")
else:
    print("not present")
s2 = {5,6,11,1,1}
print(s2)
s2.add(50)
print(s2)
s.clear()
a1 = {1,2,3,4}
a2 = {2,3,4,5,6}

#Those element which are only in one of the sets
print(a1^a2)
#A minimalistic Python Class


class Person:
    pass

p = Person()
print(type(p))
class Person:
    #Class Variable, common for all objects of the same class
    nationality = "Indian"
    
    def __init__(self,pname,clg):
        self.name = pname
        self.college = clg
    
    def sayHi(self,name):
        print("Hello "+name)
        
    def __secretMethod(self):
        print("In Secret Method of ",self.name)
    
    def introduce(self):
        print("My Name is ",self.name)
        print("I am from ",self.college)
        print("I am ",self.nationality)
        self.__secretMethod()
        
        
p = Person("Prateek","Coding Blocks")
p.sayHi("World!")
p.introduce()
#p.__secretMethod()

p2 = Person("Arnav","Micromax")
p2.introduce()
#Instance Variables vs Class Variables

class Dog:
    
    color = "Brown"
    #Common for all data member of the class 
  
    def __init__(self,breed):
        """This method accepts the breed of the dog and initialsies it"""
        self.activities = []
        self.breed = breed
    
    def addActivity(self,act):
        self.activities.append(act)
        
    def __secretActivity(self):
        print("Doing Secret Activity ",self.breed)
        
    def doActivity(self):
        """This is reg dog activities"""
        print(self.breed)
        print(self.activities)
        self.__secretActivity()
        
        
d1 = Dog("German Shepherd")
d2 = Dog("Golden Retriever")

d1.addActivity("HighJump")
d1.addActivity("Roll Over")
#d1.__secretActivity()

d2.addActivity("LowJump")
d2.addActivity("Roll Upside Down")
#d2.__secretActivity()
d1.doActivity()
d2.doActivity()

#Public vs Private method
#Inheritance in Python
class SchoolMember:
    def __init__(self,name,age):
        self.name = name
        self.age = age
        print("Init School Member: %s "%self.name)
        
    def introduce(self):
        print("Name :%s %d"%(self.name,self.age))
        

class Teacher(SchoolMember):
       
        def __init__(self,name,age,salary):
            SchoolMember.__init__(self,name,age)
            self.salary = salary
            print("Init Teacher : %s"%self.name)

        def introduce(self):
            SchoolMember.introduce(self)
            print("Salary : %d"%(self.salary))
            
class Student(SchoolMember):
    '''Represents a school student'''
    def __init__(self,name,age,marks):
        SchoolMember.__init__(self,name,age)
        self.marks = marks
        print("Init Student %s"%(self.name))
        
    def introduce(self):
        SchoolMember.introduce(self)
        print("Marks %d"%(self.marks))
    

t = Teacher("Mr.Alpha",30,45000)
t.introduce()

s = Student("Xyz",20,90)
s.introduce()
try:
    a  = input("Enter your name")
    if(len(a)<3):
        raise Exception
    
except FileNotFoundError as e:
    print("File doesnt exist. Please reupload")
    print(e)
    
except NameError as e:
    print("b is not defined")
    print(e)
    
except Exception as e:
    print("Please enter a valid name")
    print(e)
    
else:
    print("Try executed without any error")
    print("Form Submitted Successfully")
    
finally:
    print("It is always there")
s = "1+10/2+abs(-3)"

eval(s)
expression = input()
print(eval(expression))
l = ["mango","apple","banana"]
for x in enumerate(l):
    print(x)
l = []
print(all(l))
print(any(l))




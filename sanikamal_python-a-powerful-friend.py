a=[1,2,3,4]
# Assign a to a new variable b
b=a

# a and b actually refer to the same object
a.append(8)
b
my_str="Hello world"
my_str.isupper()
my_str.upper()
getattr(my_str, 'swapcase')
str="Hello "
str2="world"
print(str+str2)
s="Hello World"
s = s.replace("Hello World","Goodbye World")
s
s="Rita"
s=s.replace("R","G")
s
name=input("Enter name:")
print(name)
name=input("Enter name:")
age=int(input("Enter age:"))
job=input("Enter job:")
print(name)
print(age)
print(job)
a=None
a is None
b=6
b is not None
s = [1,2,3,4]
s[0]
s[-1]
del s[1]
s
s.append(5)
s
len(s)
min(s)
max(s)
empty_set = set()
data_scientist = set(['Python', 'R', 'SQL', 'Git','PySpark', 'SAS']) 
data_engineer = set(['Python', 'Java', 'Scala', 'Git', 'SQL', 'Hadoop','Spark'])
data_scientist
data_engineer
data_scientist = {'Python', 'R', 'SQL', 'Git','PySpark', 'SAS'}
data_engineer ={'Python', 'Java', 'Scala', 'Git', 'SQL', 'Hadoop','Spark'}
data_scientist
data_engineer
# Initialize set with values 
programmer = {'C', 'CPP', 'Java', 'Python', 'Ruby'}
programmer.add('C#')
programmer
programmer.add(['Php', 'Go'])
programmer.remove('CPP')
programmer.remove('CPP')
programmer.discard('C#')
programmer.pop()
programmer.clear()
# Initialize a set 
myskill={'c','cpp','java','python','php','html'}

for skill in myskill: 
   print(skill)
type(sorted(myskill))
myskill
print(list(set([1, 2,5, 1, 5])))
def remove_duplicates(original):
    unique = []
    [unique.append(n) for n in original if n not in unique]
    return(unique)

print(remove_duplicates([1, 2, 3, 1, 3]))
odd_num={3,5,7,11,13,19}
even_num={4,6,8,10,12}
# set built-in function union
print(odd_num.union(even_num))

# Equivalent Result 
print(odd_num | even_num)
# set built-in function intersection
print(odd_num.intersection(even_num))

# Equivalent Result 
print(odd_num & even_num)
# These sets have no elements in common so it would return True
odd_num.isdisjoint(even_num)
# Difference Operation
print(odd_num.difference(even_num))

# Equivalent Result
print(odd_num - even_num)
# Symmetric Difference Operation
print(odd_num.symmetric_difference(even_num))

# Equivalent Result
print(odd_num ^ even_num)
{skill for skill in ['C', 'C', 'CPP', 'CPP','C#']}
{skill for skill in ['git', 'C', 'sql'] if skill not in {'git', 'sql', 'CPP'}}
# Initialize a list
my_num = [1,2,3,4,5,9,77,14]

# Membership test
1 in my_num
# Initialize a set
my_num = {1,2,3,4,5,9,77,14}

# Membership test
77 in my_num
possible_skills = {'Python', 'R', 'SQL', 'Git', 'Tableau', 'SAS'}
my_skills = {'Python', 'SQL'}
my_skills.issubset(possible_skills)

# A Python program to demonstrate inheritance  
# Base or Super class. Note object in bracket.
# (Generally, object is made ancestor of all classes)
class Person:
     
    # Constructor
    def __init__(self, name):
        self.name = name
 
    # To get name
    def getName(self):
        return self.name
 
    # To check if this person is employee
    def isEmployee(self):
        return False
 
 
# Inherited or Sub class (Note Person in bracket)
class Employee(Person):
 
    # Here we return true
    def isEmployee(self):
        return True
 
# Driver code
emp = Person("Priya")  # An Object of Person
print(emp.getName(), emp.isEmployee())
 
emp = Employee("Karis") # An Object of Employee
print(emp.getName(), emp.isEmployee())
# Python example to check if a class is subclass of another
# issubclass() tells us if a class is subclass of another class.
 
class Base:
    pass   # Empty Class
 
class Derived(Base):
    pass   # Empty Class
 
# Driver Code
print(issubclass(Derived, Base))
print(issubclass(Base, Derived))
 
d = Derived()
b = Base()
 
# b is not an instance of Derived
print(isinstance(b, Derived))
 
# But d is an instance of Base
print(isinstance(d, Base))
# Python example to show working of multiple inheritance
class Base1:
    def __init__(self):
        self.str1 = "John1"
        print ("Base1")
 
class Base2:
    def __init__(self):
        self.str2 = "John2"       
        print ("Base2")
 
class Derived(Base1, Base2):
    def __init__(self):
         
        # Calling constructors of Base1
        # and Base2 classes
        Base1.__init__(self)
        Base2.__init__(self)
        print ("Derived")
         
    def printStrs(self):
        print(self.str1, self.str2)
        
 
ob = Derived()
ob.printStrs()
# Python example to show that base class members can be accessed in
# derived class using base class name
class Base:
 
    # Constructor
    def __init__(self, x):
        self.x = x    
 
class Derived(Base):
 
    # Constructor
    def __init__(self, x, y):
        Base.x = x 
        self.y = y
 
    def printXY(self):
      
       # print(self.x, self.y) will also work
       print(Base.x, self.y)
 
 
# Driver Code
d = Derived(10, 20)
d.printXY()
# Python example to show that base class members can be accessed in
# derived class using super()
class Base:
 
    # Constructor
    def __init__(self, x):
        self.x = x    
 
class Derived(Base):
 
    # Constructor
    def __init__(self, x, y):    
        super().__init__(x)
        self.y = y
 
    def printXY(self):
 
       # Note that Base.x won't work here
       # because super() is used in constructor
       print(self.x, self.y)
 
 
# Driver Code
d = Derived(10, 20)
d.printXY()
# Base Vehicle class
class Vehicle: 
    
    def __init__(self, color, manuf):
        self.color = color
        self.manuf = manuf
        self.gas = 4 # a full tank of gas
    
    def drive(self):
        if self.gas > 0:
            self.gas -= 1
            print('The {} {} goes VROOOM!'.format(self.color, self.manuf))
        else:
            print('The {} {} sputters out of gas.'.format(self.color, self.manuf))
            
# Inherits from Vehicle class           
class Car(Vehicle): 
    
    # turn the radio on
    def radio(self):    
        print("Rockin' Tunes!")

    # open the window
    def window(self):
        print('Ahhh... fresh air!')
        
# Inherits from Vehicle class            
class Motorcycle(Vehicle): 
    
    # put on motocycle helmet
    def helmet(self):
        print('Nice and safe!')
        
# Inherits from Car class
class ECar(Car):

    # an eco-friendly drive method
    def drive(self):
        print('The {} {} goes ssshhhhh!'.format(self.color, self.manuf))


# create car & motorcycle objects        
my_car = Car('red', 'Mercedes')
my_mc = Motorcycle('silver', 'Harley')        
# create and use an electric car
my_ecar = ECar('white','Nissan')

# take them out for a test drive
my_car.drive()
my_mc.drive()
my_mc.drive()
my_mc.drive()
my_mc.drive()
my_mc.drive() # out of gas
my_car.drive()

# play around with accessories
my_car.radio()
my_car.window()
my_mc.helmet()
# my_mc.window() # windows do not exist on motorcycles        
my_ecar.window()
my_ecar.radio()
my_ecar.drive()

# access the lingering gas tank
print(my_ecar.gas)

# retrieving data from the internet
import urllib.request

def main():
  # open a connection to a URL using urllib2
  webUrl = urllib.request.urlopen("http://www.google.com")
  
  # get the result code and print it
  print ("result code: " + str(webUrl.getcode()))
  
  # read the data from the URL and print it
  data = webUrl.read()
  print (data)

if __name__ == "__main__":
  main()

# A generator function that yields 1 for first time,2 second time and 3 third time 
def simple_generator(): 
    yield 1            
    yield 2            
    yield 3  
    
# Driver code to check above generator function 
for value in simple_generator():  
    print(value) 
# A Python program to demonstrate use of generator object with next()  
  
# A generator function 
def simple_generator2(): 
    yield 1
    yield 2
    yield 3

# x is a generator object 
x = simple_generator2() 
  
# Iterating over the generator object using next 
print(x.__next__())
print(x.__next__())
print(x.__next__())

# function solution
def even_integers_function(n):
    result = []
    for i in range(n):
        if i % 2 == 0:
            result.append(i)
    return result
even_integers_function(8)
# generator solution
def even_integers_generator(n):
    for i in range(n):
        if i % 2 == 0:
            yield i
even_integers_generator(10)
list(even_integers_generator(10))
#list of mixed format numbers
numbers = [7, 32, 4.5, 9.7, '3', '6']

#convert numbers to integers using expression
integers = (int(n) for n in numbers)
integers
integers.__next__()
integers.__next__()
list(integers)
even_integers = (n for n in range(10) if n%2==0)
list(even_integers)
names_list = ['Sani','Kamal','Riya','John','Mala','Rumi','Priya','Sana']

#Converts names to uppercase
uppercase_names = (name.upper() for name in names_list)

list(uppercase_names)
# too long
# reverse_uppercase = (name[::-1] for name in (name.upper() for name in names_list))

# breaking it up 
upper_case = (name.upper() for name in names_list)
reverse_uppercase = (name[::-1] for name in upper_case)
list(reverse_uppercase)
# Fibonacci Sequence Generator
def fibonacci_gen():
    a, b = 0, 1
    while True:
        yield b
        a, b = b, a + b
fib=fibonacci_gen()
fib.__next__()
fib.__next__()
fib.__next__()


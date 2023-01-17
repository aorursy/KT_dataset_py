# 1. abs() - This function returns the absolute value of a number.
# syntax = abs(number)
inp = input("Please enter a number: ")
y = abs(int(inp))
print("The absolute value of", inp, "is", y)
#2. all() - This function returns true if all items in an iterable object are true
# syntax = all(iterable)
# Example 1
L1 = [0,1,1]
x= all(L1) # Here the given list has 0 which is false. Since there is one false, output is false.
print(x)
#2. all() - This function returns true if all items in an iterable object are true
# syntax = all(iterable)
# Example 2
tup1 = (1,1,1)
x=all(tup1) # Here for the given tuple has all 1's which are true. Hence output is True.
print(x)
#2. all() - This function returns true if all items in an iterable object are true
# syntax = all(iterable)
# Example 3
set1 = {True,False,True}
x= all(set1) # Here for the given set, one element is false. Hence output is false
print(x)
#2. all() - This function returns true if all items in an iterable object are true
# syntax = all(iterable)
# Example 4
dict1 = {0:"Apple",1:"Orange"}
x=all(dict1) # Here in the given dictionary, the first key is 0 which is false. Hence the output is false.
print(x)
#3. any() - This function returns true if any item in an iterable object is true
# syntax = any(iterable)
L1 = [0,0,1]
x = any(L1)
print(x)
#4. ascii() - This function returns a readable version of any object (Lists, Tuples,strings etc.). All non ascii characters will be replaced by \xe5.
# You can refer the list of non-ascii characters from the link https://terpconnect.umd.edu/~zben/Web/CharSet/htmlchars.html
# syntax = ascii(object)
x = "My nåme is Råvitejå" # Here å is the non ascii character.
y = ascii(x)
print(y)
#5. bin() - This function returns the binary version of a number. The output has always prefix of "0b".
# syntax = bin(number)
x = bin(24) 
print(x)
#6. bool() - This function returns the boolean value of the specified object. 
# This function returns false for objects like [],{},(), False, 0, None. For other values the function returns true.
# syntax = bool(object)
x = bool("Ravi")
print(x)
#7. bytearray() - This function returns an array of bytes.
# The output always has prefix of b. Output will be in an array.
# syntax = bytearray(x,encoding,error). If x is an integer,an empty bytearray object of specified size will be created. If it is a string, we need to specify the encoding of source.
x = bytearray(5)
print(x)
#8. bytes() - This function returns a bytes object.
# The output always has prefix of b.
# syntax = bytes(x,encoding,error). If x is an integer,an empty bytearray object of specified size will be created. If it is a string, we need to specify the encoding of source.
x = bytes(4)
print(x)
#9. callable() - This function returns True if it is callable else false.
# syntax = callable(object)
# Example 1
def x():
    i = 5
print(callable(x))
#9. callable() - This function returns True if it is callable else false.
# syntax = callable(object)
# Example 2
i = 5
print(callable(i))
#10. chr() - This function returs a character from the specified unicode code.
# Unicode characters table can be referred from the link https://www.rapidtables.com/code/text/unicode-characters.html
# syntax = chr(number)
x = chr(98)
print(x)
#11. classmethod() - This function converts a method into class method.
# syntax = classmethod(function)
class Person:
    age = 25
    def printAge(cls):
        print('The age is:', cls.age)

# create printAge class method
Person.printAge = classmethod(Person.printAge)
Person.printAge()
#12. compile() - This function returns the specified source as an object and ready to be executed.
# Its format is compile(source,filename, mode, flag, dont_inherit, optimize). Flag, dont_inherit and optimize are optional.
# Legal values of mode are : eval - if the source is single expression ; exec - if the source is a block of statements; single - if the source is a single interactive statement.
# syntax = compile(source,filename,mode)
# Example 1
x = compile("print(5)", "test","eval")
exec(x)
#12. compile() - This function returns the specified source as an object and ready to be executed.
# Its format is compile(source,filename, mode, flag, dont_inherit, optimize). Flag, dont_inherit and optimize are optional.
# Legal values of mode are : eval - if the source is single expression ; exec - if the source is a block of statements; single - if the source is a single interactive statement.
# syntax = compile(source,filename,mode)
# Example 2
x = compile("print(5)\nprint(9)","test","exec")
exec(x)
#13. complex() - This functions returns a complex number
# Here j is the imaginary part indicator.
# syntax = comple(real,imaginary)
x = complex(2,4)
print(x)
#14. delattr() - This function deletes the specified attribute (property or method) from the specified object.
# syntax = delattr(object,attribute)
class Person: # Creating a class
    name = "Ravi"
    age = 25
    country = "India"
delattr(Person,"age") # Removing age attribute from the class Person.
print(Person.age) # Since age attribute is removed, hence we print it, it will show error.
#15. dict() - This function returns a dictionary (Array)
# dict() function creates a dictionary.
# syntax = dict(Key word arguments)
x = dict(Name = "Ravi", Age = 25, Country = "India")
print(x)
#16. dir() - This function returns a list of specified object's properties and methods without the values.
# syntax = dir(object)
class Person:
  name = "Ravi"
  age = 25
  country = "India"
print(dir(Person))
#17. divmod() - This function returns the quotient and remainder when argument 1 is divided by argument 2.
# syntax = divmod(dividend,divisor)
a = input("Please enter the first number: ")
b = input("Please enter the second number: ")
divmod(int(a),int(b))
#18. enumerate() - This function converts the tuple into enumerate object i.e. complete ordered listing of all the items in the collection.
# Here numbering starts from 0.
# syntax = enumerate(iterable,start)
x = ("Ravi","teja","Raviteja")
y = enumerate(x)
print(list(y))
#19. eval() - This function evaluates the specified expression and if it is a legal python statement, it will be executed.
# The initial parameter must be a string.
# syntax = eval(expression)
x = "print(55)"
eval(x)
#20. exec() - This function executes a specified python code.
# syntax = exec(object)
name = "Ravi"
print(name)
exec(name)
#21. filter() - This function is used to filter items in an iterable object.
# syntax = filter(function, array)
age = [12,14,32,18,6,45]
def adult(x): # Defining a function named adult which gives true if age is greater than or equal to 18 and false if age is less than 18.
    if x < 18:
        return False
    else:
        return True
adults = filter(adult,age)
for x in adults: # This will print the age which is greater than or equal to 18.
    print(x)
#22. float() - This function converts the value into floating point number.
# syntax = float(value)
inp = input("Please enter the number: ")
x = float(inp) # The input will be stored in form of string. This function will convert it from string to float.
print(x)
#23 format() - This function converts the given value into a specified format.
# syntax = format(value,format)
x = format(0.15,"%") # Converting the number into % format.
print(x)
#24. frozenset() - This function returns an unchangeable frozen set.
# syntax = frozenset(iterable object)
# If we try to change the value of frozenset item, it will show error.
x = ["Ravi", "teja", "Krishna"]
y = frozenset(x)
print(y)
#25. getattr() - This function returns the value of specified attribute.
# syntax = getattr(object,attribute)
class Student:
    Name = "Ravi"
    Age = 25
    Country = "India"
x = getattr(Student,"Age") # Returns the value of attribute age.
print(x)

#26. globals() - This function returns the global symbol table as a dictionary.
#syntax = globals()
x = globals()
print(x)
#27. hasattr() - This function returns true if the specified object has specified attribute.
# syntax = hasattr(object,attribute)
class Student:
    Name = "Ravi"
    Age = 25
    Country = "India"
x = hasattr(Student,"Age") # Since age attribute is present in the object Student. Hence this returns true.
print(x)
#28. hash() - This function returns the hash value of the specified object.
# syntax = hash(object)
x = hash("Ravi")
print(x)
#29. help() - This function is used to display the documentation of modules, functions, classes, keywords etc.
# syntax = help(object)
x = help(print)
print(x)
#30. hex() - This function converts a value into hexadecimal value
# syntax = hex(number)
x = hex(27)
print(x)
#31. id() - This function returns the unique id of the object. Id will be different for the same program executing multiple times.
# syntax = id(object)
x = [20,22,46]
y = id(x)
print(y)
#32. input() - This function allows user input.
# syntax = input(prompt)
inp = input("Please enter your name: ")
print("Hello,", inp)
#33. int() - This function converts the given number into integer. The input is always stored as string.
# syntax = int(object)
inp = input("Please enter the number: ")
x = int(inp)
print(x)
#34. isinstance() - This function returns true if the specified object is of specified type.
# syntax = isinstance(object,type)
x = isinstance("Ravi", str) # Ravi is a string. Hence the output is true.
print(x)
#35. issubclass() - This function returns true if the specified object is the subclass of the specified object.
# syntax = issubclass(object,subclass)
class MyAge:
    age = 25
class Student(MyAge): #MyAge is the subclass of Student. Hence output is True.
    name = "Ravi"
    age = MyAge
x = issubclass(Student,MyAge)
print(x)
    
#36. iter() - This function returns an iterator object.
# syntax = iter(object)
x = iter(["Mango","Banana","Apple"])
print(next(x)) # This will print first item in the list
print(next(x)) # This will print second item in the list
print(next(x)) # This will print third item in the list
#37. len() - This function returns the length of the object.
# syntax = len(object)
L1 = [2,4,55,31,9]
len(L1)
#38. list() - This function creates a list object.
# syntax = list(iterable)
x = ("Ravi","Teja","Raviteja")
y = list(x)
print(y)
#39. locals() - This function returns the local symbol table as dictionary.
# syntax = locals()
x = locals()
print(x)
#40. map() - This function executes a specified function for each item in an iterable.
#syntax = map(function,iterables)
def length(n):
    return len(n)
y = map(length,("Apple","banana","orange")) # 
print(list(y))
#41. max() - This function returns the largest item in an iterable.
# syntax = max(x1,x2,x3,...)
x = max(45,48)
print(x)
#42. memoryview() - This function returns memory view object from a specified object.
#syntax = memoryview(object)
x = memoryview(b"Hello")
print(x)
print(x[0]) #return the Unicode of the first character
print(x[1])#return the Unicode of the second character
#43. min() - This function returns the smallest item in an iterable.
# syntax = min(n1,n2,n3,...)
x = min(10,20)
print(x)
#44. next() - This function returns the next item in an iterable.
# syntax = next(iterable)
L1 = iter(["Mango","Apple","Banana"]) #Creating iterable
x = next(L1)
print(x)
x = next(L1)
print(x)
x = next(L1)
print(x)
#45. object() - This function returns a new empty object.
# syntax = object()
x = object()
#46. oct() - This function converts a number into an octal value
# syntax = oct(int)
x = 12
y = oct(x)
print(y)
#47. open() - This function opens a file and returns it as a file object.
# syntax(file,mode)
# In mode r = read ; a = append; w = write; x = create
f = open("demofile.txt", "r")
print(f.read()) # This will open the demofile.txt and it will read.
#48. ord() - This function returns the number representing the unicode of a specified character.
# syntax = ord(character)
x = ord("h")
print(x)
#49. pow() - This function returns the value of x to the power of y (x^y)
#syntax = pow(x,y)
x = pow(2,3)
print(x)
#50. print() - This function prints the message to the screen.
# syntax = print(objects)
print("Hello World")
#51. property() - This function creates property of a class.
# syntax = property(class1,class2)
class Person:
    def init(self, name):
        self._name = name

    def get_name(self):
        print('Getting name')
        return self._name

    def set_name(self, value):
        print('Setting name to ' + value)
        self._name = value

    def del_name(self):
        print('Deleting name')
        del self._name

    # Set property to use get_name, set_name
    # and del_name methods
    name = property(get_name, set_name, del_name, 'Name property')

p = Person('Adam')
print(p.name)
p.name = 'John'
del p.name
#52. range() - This function returns sequence of numbers starting with 0 and increments by 1 and stops before a specified number.
# syntax = range(stop)
# Bydefault it starts with 0 and ends with (stop value-1).
x = range(8)
for y in x:
    print(y,end=" ")
#53. repr() - This function returns a printable representation of the given object.
# syntax = repr(object)
x = "Ravi"
y = repr(x)
print(y)
#54. reversed() - This function returns a reversed iterator object.
# syntax = reversed(object)
x = ["Apple","Mango","Banana"]
y = reversed(x)
for z in y:
    print(z,end=" ")
#55. round() - This function returns a floating point number which is rounded to the specific number of decimals.
# syntax = round(number,digits)
x = 4.28567
y = round(x,2) # Here x is rounded to 2 decimals. 
print(y)
#56. set() - This function creates a set object.
# syntax = set(iterable)
x = set(("Apple","Banana","Orange"))
print(x)
#57. setattr() - This function sets the value of specified attribute of the specified object.
# syntax = setattr(object,attribute,value)
class Student:
    name = "Ravi"
    age = 25
    country = "India"
setattr(Student,"age",35) # This will set the age from 25 to 35.
x = getattr(Student,"age") # This will return the set value.
print(x)
#58. slice() - This function is used to specify how to slice a sequence. We can specify start,end of the slice.
#syntax = slice(start,end,step)
# Start and step are optional. By default start is 0 and step is 1.
x = ("a","b","c","d","e","f")
y = slice(2,5,1)
print(x[y])
#59. sorted() - This function returns a sorted list of specified iterable object.
# syntax = sorted(iterable)
x = (5,3,6,8,4,2)
y = sorted(x)
print(y)
#60. staticmethod() - This function returns a static method for teh function.
# syntax = staticmethod(function)
class Mathematics:
    def addNumbers(x, y):
        return x + y

# create addNumbers static method
Mathematics.addNumbers = staticmethod(Mathematics.addNumbers)

print('The sum is:', Mathematics.addNumbers(5, 10))
#61. str() - This function converts a specified value to string.
# syntax = str(object)
x = str(5) # Converted type from int to str.
print(x)
type(x)
#62. sum() - This function returns the sum of all items in an iterable.
# syntax = sum(iterable)
x = (1,2,3,4,5)
y = sum(x)
print(y)
#63. super() - This function returns an object that will reflect the parent class.
# syntax = super()
class Parent:
  def __init__(self, txt):
    self.message = txt

  def printmessage(self):
    print(self.message)

class Child(Parent):
  def __init__(self, txt):
    super().__init__(txt)

x = Child("Hello, and welcome!")

x.printmessage()
#64. tuple() - This function creates a tuple object.
# syntax = tuple(iterable)
x = ("Apple","Banana","Orange")
y = tuple(x)
print(y)
#65. type() - This function will return the type of specified object.
# syntax = type(object)
x = "ravi"
type(x)
#66. vars() - This function returns the dict attribute of the object.
# syntax = vars(object)
class Student:
    name = "Ravi"
    age = 25
    country = "India"
x = vars(Student)
print(x)
#67. zip() - This function returns a zip object which is an iterator of tuples. 
# If the passed iterators have different lengths, the iterator with least items decide the length of new iterator.
# syntax = zip(iterator 1, iterator 2, iterator 3,...)
a = ("Ravi", "sagar", "rama")
b = ("Teja", "sachin", "Krishna", "Venky")
x = zip(a,b)
print(tuple(x)) # In output, Venky is neglected since a has length 3 and b has length 4.
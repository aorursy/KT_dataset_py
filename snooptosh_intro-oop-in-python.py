# Create a class named MyClass, with a property named x:

class MyClass:

    x = 5
# Now we can use the class named MyClass to create objects:

# Create an object named p1, and print the value of x:

p1 = MyClass()

print(p1.x)

# Create a class named Person, use the __init__() function to assign values for name and age:



class Person:

    def __init__(self, name, age):

        self.name = name

        self.age = age



p1 = Person("John", 36)



print(p1.name)

print(p1.age)

# Insert a function that prints a greeting, and execute it on the p1 object:



class Person:

    def __init__(self, name, age):

        self.name = name

        self.age = age

        

    def myfunc(self):

        print(f"Hello, my name is {self.name}")

        



p1 = Person("Mike", 27)

p1.myfunc()
class Person:

    def __init__(mysillyobject, name, age):

        mysillyobject.name = name

        mysillyobject.age = age



    def myfunc(abc):

        print("Hello my name is " + abc.name)



p1 = Person("John", 36)

p1.myfunc()
class Person:

    def __init__(self, fname, lname):

        self.firstname = fname

        self.lastname = lname

        

    def printname(self):

        print(f"My name is {self.firstname} {self.lastname}")

        

x = Person("Mike", "Olsen")

x.printname()
# Create a class named Student, which will inherit the properties and methods from the Person class:



class Student(Person):

    pass



# Note: Use the pass keyword when you do not want to add any other properties or methods to the class.
x = Student("Mike", "Olsen")

x.printname()

# class Student(Person):

#     def __init__(self, fname, lname):

        # add properties here

#         self.firstname = fname

#         self.lastname = lname
class Student(Person):

    def __init__(self, fname, lname):

        Person.__init__(self, fname, lname)
class Student(Person):

    def __init__(self, fname, lname):

        super().__init__(fname, lname)
class Student(Person):

    def __init__(self, fname, lname, year):

        super().__init__(fname, lname)

        self.graduationyear = year

    

    # add a method called 'welcome' to the 'Student' class

    def welcome(self):

        print("Welcome!", self.firstname, self.lastname, "to the class of", self.graduationyear)    

        

p1 = Student("Mike", "Olsen", 2015)

p1.welcome()
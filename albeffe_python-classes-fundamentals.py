class Class_A:
    pass
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    # Objects can also contain methods. Methods in objects are functions that belong to the object.
    def my_method(self):
        print("Hello my name is " + self.name)
p1 = Person("John", 36)

print(p1.name)
print(p1.age)
p1.my_method()

# You can modify properties on objects
p1.age = 40

# You can delete properties of objects or objects themselves by using the del keyword
del p1.age
del p1
class Student(Person):
    pass


x = Student("Mike", "Olsen")
x.my_method()
class Student(Person):
    def __init__(self, fname, lname):
        Person.__init__(self, fname, lname)
class Student(Person):
    def __init__(self, fname, lname):
        super().__init__(fname, lname)
        self.student_property = 2019
class Base:

    # Declaring public method
    def fun(self):
        print("Public method")

        # Declaring private method

    def __fun(self):
        print("Private method")

    # Creating a derived class


class Derived(Base):
    def __init__(self):
        # Calling constructor of
        # Base class
        Base.__init__(self)

    def call_public(self):
        # Calling public method of base class
        print("\nInside derived class")
        self.fun()

    def call_private(self):
        # Calling private method of base class
        self.__fun()
obj1 = Base()

# Calling public method
obj1.fun()

obj2 = Derived()
obj2.call_public()

# Uncommenting obj1.__fun() will
# raise an AttributeError

# Uncommenting obj2.call_private()
# will also raise an AttributeError
class A:

    # Declaring public method 
    def fun(self):
        print("Public method")

        # Declaring private method

    def __fun(self):
        print("Private method")

        # Calling private method via

    # another method
    def Help(self):
        self.fun()
        self.__fun()

    # Driver's code


obj = A()
obj.Help()
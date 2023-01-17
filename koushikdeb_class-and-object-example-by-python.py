class Computer:
    def config(self):
        print("i5, 1Tb computer")


comp1 = Computer()
comp2 = Computer()

Computer.config(comp1)
Computer.config(comp2)

comp1.config()
comp2.config()
class Computer:
    def __init__(self):
        print("I am init method")

    def config(self):
        print("i5, 1Tb computer")


comp1 = Computer()
comp2 = Computer()

comp1.config()
comp2.config()
class Computer:
    def __init__(self, cpu, ram):
        self.cpu = cpu
        self.ram = ram

    def config(self):
        print("Config is :", self.cpu, self.ram)


c1 = Computer('i5', '4gb')
c2 = Computer('Ryzen', '8gb')

c1.config()
c2.config()
class Person:
    def __init__(self): # this method is known as constructor
        self.name = "Navin"
        self.age = 28

    def update(self):
        self.age = 35

    def compare(self, other):
        if self.age == other.age:
            return True
        else:
            return False


p1 = Person()
p2 = Person()

p1.name = "Happy"
p1.age = 30

p1.update()

print("Name : ", p1.name)  # Return the address of Computer
print("Age: ", p1.age)  # Return the address of Computer
print("Name : ", p2.name)  # Return the address of Computer
if p1.compare(p2):
    print("They are same")
else:
    print("They are not same")
class Car:
    """
    Variable define outside the init method is called Class variable or Static Variable
    """
    wheels = 4

    def __init__(self):
        """
        Varible define in the init mathod is called instance varible
        """
        self.mil = 10
        self.com = "BMW"


c1 = Car()
c2 = Car()

c1.mil = 8

Car.wheels = 6

print(c1.mil, c1.wheels)
print(c2.mil, c2.wheels)
class Student:

    school = "Telusko" # class variable

    def __init__(self, m1, m2, m3):
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3

    def avg(self):
        """
        This is intance method because its work with object like self method
        """
        return (self.m1 + self.m2 + self.m3) / 3

    def getm1(self):
        """
        Accessor : It's used to access the value or variable or method
        Mutators : It's used to change the value of variable or method
        """
        return self.m1

    def setm1(self, value):
        self.m1 = value

    @classmethod
    def getschoolname(cls):
        return cls.school

    @staticmethod
    def info():
        print("This is static method")


s1 = Student(25, 40, 55)
s2 = Student(50, 60, 70)
s1.avg(), s2.avg()
s1.getschoolname()
print(Student.getschoolname())
s1.info()
Student.info()
class student:  ## Outer Class
    def __init__(self, name, rollno):
        self.name = name
        self.rollno = rollno
        self.lap = self.Laptop(
        )  #we can create this object inside the class and also outside the class

    def show(self):
        print(self.name, self.rollno)
        self.lap.show()

    class Laptop:  ## Inner Class
        def __init__(self):
            self.brand = "HP"
            self.cpu = "i5"
            self.ram = 8

        def show(self):
            print(self.brand, self.cpu, self.ram)


s1 = student("Navin", 2)
s2 = student("Ashish", 3)

print(s1.name, s1.rollno)

s1.show()
s1.lap.brand

lap1 = s1.lap
lap2 = s2.lap
print(id(lap1))
print(id(lap2))
# create Laptop object outside the class

laptop1 = student.Laptop()
print(laptop1)
class A:
    def feature1(self):
        print("feature1 is working")

    def feature2(self):
        print("feature2 is working")
        
class B: ## single level inheritance
    def feature3(self):
        print("feature3 is working")

    def feature4(self):
        print("feature4 is working")
    
class C(B): #hierarchical inheritance
    def feature5(self):
        print("feature5 is working")

class D(A,B): #multiple inheritance
    def feature6(self):
        print("feature6 is working")
    
a1 = A()
b1 = B()
a1.feature1()
a1.feature2()
b1.feature3()
b1.feature4()
c1 = C()
d1 = D()
c1.feature3()
d1.feature6()
class A:
    def __init__(Self):
        print("It's A init")

    def feature1(self):
        print("feature1 is working")

    def feature2(self):
        print("feature2 is working")


class B(A):  ## single level inheritance
    def __init__(self):
        super().__init__()
        print("It's B init")

    def feature3(self):
        print("feature3 is working")

    def feature4(self):
        print("feature4 is working")
a2 = A()
a3 = B()
class A:
    def __init__(Self):
        print("It's A init")

    def feature1(self):
        print("feature1-A is working")

    def feature2(self):
        print("feature2 is working")


class B:  ## single level inheritance
    def __init__(self):
        print("It's B init")

    def feature1(self):
        print("feature1-B is working")

    def feature4(self):
        print("feature4 is working")


class C(A, B):
    def __init__(self):
        super().__init__()
        print("It's C init")
        
    def feat(self):
        super().feature2()


cam = C()
cam.feat()
class IDE:
    def execute(self):
        print("Compiling")
        print("Running")
        

class Laptop:
    def code(self,ide):
        ide.execute()
        
        
class Editor:
    def execute(self):
        print("Spell Check")
        print("Compiling")
        print("Running")
        
        
    
ide = Editor()
lap1 = Laptop()
lap1.code(ide)
class Student:
    def __init__(self, m1, m2, m3):
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3

    def __add__(self, other):
        m1 = self.m1 + other.m1
        m2 = self.m2 + other.m2
        m3 = self.m3 + other.m3
        s3 = Student(m1, m2, m3)
        return s3
    
    def __gt__(self, other):
        r1 = self.m1 + self.m2
        r2 = other.m1 + self.m2
        if r1 > r2:
            return True
        else:
            return False
        
    def __str__(self):
        return '{} {}'.format(self.m1, self.m2)
        


s1 = Student(50, 50, 25)
s2 = Student(40, 70, 20)
s3 = Student(25, 40, 20)

s4 = s1 + s2 + s3

print(s4.m1)

if s1 > s2:
    print("s1 wins")
else:
    print("s2 wins")
print(s4.__str__())
# this is example of method overloading
class Student:
    def __init__(self, m1, m2):
        self.m1 = m1
        self.m2 = m2
    
    def sum(self, a =None,b = None, c = None):
        s = 0
        if a!=None and b!=None and c!=None:
            s = a + b + c 
        elif a!=None and b!=None:
            s = a + b
        else: 
            s = a
        return s

s1 = Student(50, 50)

s1.sum(1)
# this is example of method overriding

class A:
    def show(self):
        print("I have nokia 1100 phone")
        
class B(A):
    def show(self):
        print("I have MotoG3")


abc = B()
abc.show()
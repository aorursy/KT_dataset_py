class Class:
    def __init__(self, attribute):
        print("Object created")
        self.attribute = attribute    # Variable
        
    def method(self):                 # Function
        print("Method")
# Create an object from the class
obj = Class("Attribute")

# Calling a method
obj.method()

# Accessing a attribute 
print(obj.attribute)
class Fruit:
    def __init__(self, name, colour, shape):
        self.name = name
        self.colour = colour
        self.shape = shape
        
    def describe(self):
        print(f"{self.name} is a {self.shape} {self.colour} coloured fruit")
        
    def __del__(self):
        print("Deleted")
apple = Fruit("Apple", "red", "round")
apple.describe()
banana = Fruit("Banana", "yellow", "long elongated")
banana.describe()
# Deleting an object
del apple
class A:
    def __init__(self):
        self.a = 1
    
    def method1(self):
        print(self.a)
        

class B(A):
    def __init__(self):
        super().__init__()
        self.b = 3
        
    def method2(self):
        print(self.b)
    
    def call_method1(self):
        self.method1()
    
a = A()
a.method1()
b = B()
b.method2()
b.call_method1()
class A:
    def __init__(self):
        self.a = 10
        
    def method(self):
        print(self.a * 10)
        
class B(A):
    def __init__(self):
        super().__init__()
    
    def method(self):
        print(self.a * 20)
        
a = A()
b = B()

a.method()
b.method()
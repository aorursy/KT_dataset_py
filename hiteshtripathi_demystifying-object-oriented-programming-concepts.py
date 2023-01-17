''' 

    Creating a Class named "Car"

    

    Class Varibales = 'feul_type' ----- Same across all car class object

    Object Variable = 'model','color','mileage'--Different for different car objects

    Object Methods/Attributes = 'description','max_speed'



'''

class Car:

    

    'Common for all the objects of Car Class'

    fuel_type = 'Petrol'       ## Class Attribute

    

    def __init__(self, model,color,mileage): ## the default method which will be called 

                                             #autmatically when the object of this calls is called.

        

        self.model_name = model ## object attribute

        self.color = color     ## object attribute

        self.mileage = mileage ## object attribute

        print("In init method")

        

    

    def description(self):

        print("The {} gives the mileage of {} kmpl".format(self.model_name,self.mileage))

        

    

    def max_speed(self,speed):

        return("The {} runs at the maximum speed of {}kmphr".format(self.model_name,speed))

        
## Create an object of Class Car

obj1 = Car("Audi A6","Red",17.01)



## Call the methods and variables defined for this object from Car class.

obj1.description()

print(obj1.max_speed(233))

print(obj1.color)



## Call the Car Class variable.

print(Car.fuel_type)



## Create another object of Class car.

obj2 = Car("Audi A8","Black",19)

obj2.max_speed(800)
print("Address of obj1",id(obj1))

print("Address of obj2",id(obj2))
print(type(obj1))  ## User-defined Class. So "obj1" is the object of "Car" Class.

a = 5

print(type(a))     ## In-built Class. So "a" is the object of "int" Class.
class Car:

    

    'Common for all the objects of Car Class'

    fuel_type = 'Petrol'## Class Attribute

    

    def __init__(self, model,color,mileage): 

        

        self.model_name = model ## object attribute

        self.color = color     ## object attribute

        self.mileage = mileage ## object attribute

        print("In init method")     
class Car:

    

    'Common for all the objects of Car Class'

    fuel_type = 'Petrol'## Class Attribute

    

    def __init__(self, model,color,mileage): ## the default method which will be called autmatically 

                                             ## when the object of this calls is called.

        

        self.model_name = model     ## object attribute

        self.color = color     ## object attribute

        self.mileage = mileage ## object attribute

        print("In init method")

        

    

    def description(self):

        print("The {} gives the mileage of {} kmpl".format(self.model_name,self.mileage))

        

    

    def max_speed(self,speed):

        return("The {} runs at the maximum speed of {}kmphr".format(self.model_name,speed))



    

obj1 = Car("Audi A8","Black",19) 

print(Car.max_speed(obj1,500)) ### Calling the "max_speed" method with obj1 as a parameter which in 

                               ### default case is done by the self keyword.





print(obj1.max_speed(800)) ## Calling the method with single argument
class Car:

    

    'Common for all the objects of Car Class'

    fuel_type = 'Petrol'## Class Attribute

    

    def __init__(self, model,color,mileage): ## the default method which will be called autmatically 

                                             ## when the object of this calls is called.

        

        self.model_name = model ## object attribute

        self.color = color     ## object attribute

        self.mileage = mileage ## object attribute

        

    

    def description(self):

        print("The {} gives the mileage of {} kmpl".format(self.model_name,self.mileage))

        

    

    def max_speed(self,speed):

        return("The {} runs at the maximum speed of {}kmphr".format(self.model_name,speed))



    def compare(self,other):

        if self.mileage == other.mileage:

            return True

        else:

            False



car1 = Car("Audi A8","Blue",18)

car2 = Car("Audi A6","Red",19)



if car1.compare(car2):

    print("Same mileage")

else:

    print("Difference in mileage")
class Car:

    

    'Common for all the objects of Car Class'

    fuel_type = 'Petrol'## Class Attribute

    

    def __init__(self, model,color,mileage): ## the default method which will be called autmatically 

                                             ## when the object of this calls is called.

        

        self.model_name = model ## object attribute

        self.color = color     ## object attribute

        self.mileage = mileage ## object attribute



    def get_mileage(self): ### ------> The Accessor Method as it is feteching the values

        return(self.mileage)

    

    def set_mileage(self,value): ### -----> The Mutators method as it is modifying the existing Value

        self.mileage = value

        return(self.mileage)

    

    @classmethod  ## ---> We must indicate the "decorator" here which is indicating it is a Class method

    def info(cls): ## ---> This is a Class Method example

        

        return(cls.fuel_type)

    

    @staticmethod   ## ----> We must use the decorator to define that it is a "static method"

    def about_car():

        print("This is all about Audi Cars")

    

car1 = Car("Audi A6","Red",18)



print(car1.get_mileage())

print(car1.set_mileage(19))



Car.about_car() ## ---> This is how we call the "Static methods" using the class names



print(Car.info()) ## --> This is how we call the "Call methods"
class Car:  ## Parent class



    def __init__(self, name, mileage):

        self.name = name 

        self.mileage = mileage 



    def description(self):                

        return f"The {self.name} car gives the mileage of {self.mileage}km/l"



class BMW(Car): ## Child Class

    pass



class Audi(Car):     ## Child class

    def audi_desc(self):

        return "This is the description method of class Audi."
obj1 = BMW("BMW 7-series",39.53)

print(obj1.description())



obj2 = Audi("Audi A8 L",14)

print(obj2.description())

print(obj2.audi_desc())
class car:



    def __init__(self, name, mileage):

        self._name = name  #protected variable

        self.mileage = mileage 



    def description(self):                

        return f"The {self._name} car gives the mileage of {self.mileage}km/l"
obj = car("BMW 7-series",39.53)



#accessing protected variable via class method 

print(obj.description())



#accessing protected variable directly from outside

print(obj._name)

print(obj.mileage)
class Car:



    def __init__(self, name, mileage):

        self.__name = name              #private variable        

        self.mileage = mileage 



    def description(self):                

        return f"The {self.__name} car gives the mileage of {self.mileage}km/l"

obj = Car("BMW 7-series",39.53)



#accessing private variable via class method 

print(obj.description())



#accessing private variable directly from outside

print(obj.mileage)

print(obj.__name)
class Car:



    def __init__(self, name, mileage):

        self.__name = name  #private variable        

        self.mileage = mileage 



    def description(self):                

        return f"The {self.__name} car gives the mileage of {self.mileage}km/l"

obj = Car("BMW 7-series",39.53)



#accessing private variable via class method 

print(obj.description())



#accessing private variable directly from outside

print(obj.mileage)

print(obj._Car__name)      #mangled name

class Audi:

    def description(self):

        print("This the description function of class AUDI.")



class BMW:

    def description(self):

        print("This the description function of class BMW.")
audi = Audi()

bmw = BMW()

for car in (audi,bmw):

    car.description()
from abc import ABC, abstractmethod 

  

class Polygon(ABC): 

  

    # abstract method 

    def noofsides(self): 

        pass



class Triangle(Polygon): 

  

    # overriding abstract method 

    def noofsides(self): 

        print("I have 3 sides") 



        

class Pentagon(Polygon): 

  

    # overriding abstract method 

    def noofsides(self): 

        print("I have 5 sides") 

        

class Hexagon(Polygon): 

  

    # overriding abstract method 

    def noofsides(self): 

        print("I have 6 sides") 

        

        

class Quadrilateral(Polygon): 

  

    # overriding abstract method 

    def noofsides(self): 

        print("I have 4 sides") 

  

# Driver code 

R = Triangle() 

R.noofsides() 

  

K = Quadrilateral() 

K.noofsides() 

  

R = Pentagon() 

R.noofsides() 

  

K = Hexagon() 

K.noofsides() 
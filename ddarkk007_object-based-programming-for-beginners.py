import datetime

import time
"""

As we have seen, we have created one of our employees, but if the number of these employees is 200, 

it will take a long time to create them one by one, and object-oriented programming will help us here.

"""

employee1_name = "David"

employee1_surname = "Alex"

employee1_age = 19

employee1_position = "Marketing"
#Create a class



class EmployeeList:

  #attribute = age, phone-number, adress, name and surname

  #behaviour = capabilities

  pass
#attribute

# Note: Normally coding is not done in this way, but in general, it is written in this way to repeat and gain knowledge about the subject.

class SalesMan:

  company = "XYZ"

  age = 20

  gender = "MALE"



new_salesman1 = SalesMan()

print(new_salesman1.gender)

print(new_salesman1.company)

print(new_salesman1.age)

print(" ")



#Changing the data

new_salesman1.company = "ZYX"

print(new_salesman1.company)
#methods



#Create a new class



class Square:

  edge = 10 #meter



  def areaSqaure(self):

    area = self.edge * self.edge

    return area

    

square1 = Square()

print(square1)

print(square1.edge)

print(square1.areaSqaure())
# methods vs functions



#Create a new class



class StudentsList:

  age = 17

  exam_score = 70



  #methods

  def ageExamScoreRatio(self):

    ratio = print(self.age / self.exam_score)

    return ratio



#functions

def ageExamScoreRatio2(age1, exam_score1):

  return print(age1 / exam_score1)

  



student_1 = StudentsList()

print("Method:")

student_1.ageExamScoreRatio()

print(" ")

print("Function:")

ageExamScoreRatio2(40,80)
#Constructor



class Animals:



  def __init__(self, animal, animal_age):

    self.animal = animal

    self.animal_age = animal_age

  

  def getAge(self):

    return self.animal_age

  

  def getAnimal(self):

    return self.animal

  

animal1 = Animals("dog", 7)

animal2 = Animals("cat", 4)



print(animal1.getAge())

print(animal2.getAnimal())
#calculator project



class CalculatorProject:



  def __init__(self, value1, value2):

    self.value1 = value1

    self.value2 = value2

  

  def valueAddition(self):

    return self.value1 + self.value2

  

  def valueMultiply(self):

    return self.value1 * self.value2

  

  def valueDivision(self):

    return self.value1 / self.value2

  

  def valueSubtraction(self):

    return self.value1 - self.value2



# In its simplest form

v1 = 100

v2 = 20

c1 = CalculatorProject(v1, v2)



print(c1.valueAddition())

print(c1.valueMultiply())

print("------------------")

ct1 = c1.valueAddition()

ct2 = c1.valueMultiply()

print("Addition: {} , Multiply: {}".format(ct1, ct2))
# Improved version



"""

class CalculatorProject:



  def __init__(self, value1, value2):

    self.value1 = value1

    self.value2 = value2

  

  def valueAddition(self):

    return self.value1 + self.value2

  

  def valueMultiply(self):

    return self.value1 * self.value2

"""

print("Choose to add (1) or to multiply (2) or devision(3) or subtraction(4)")

selection = int(input("Select 1 or 2 or 3 or 4"))



v3 = int(input("first value"))

v4 = int(input("second value"))

c2 = CalculatorProject(v3,v4)



if selection == 1:

  print("Addition:", c2.valueAddition())

elif selection == 2:

  print("Multiply:", c2.valueMultiply())

elif selection == 3:

  print("Division:", c2.valueDivision())

elif selection == 4:

  print("Subtraction:", c2.valueSubtraction())

else:

  print(" ")



#finish
#Encapsulation

#restricting methods and direct access to variables



class BankAccount:



  def __init__(self, name, money, address):

    self.name = name #global variable

    self.money = money #private variable1!

    self.address = address #gloabal variable



person1 = BankAccount("john", 400, "USA")

person2 = BankAccount("messi", 600, "Canada")
person1.name
#Bank accounts are secure, but we have changed this value, now let's prevent it from being changed.



person1.money = person1.money + person2.money 

person2.money = 0



print(person1.money)

print(person2.money)
#Private variable!



class BankAccount2:



  def __init__(self, name1, money1, address1):

    self.name1 = name1 #global variable

    self.__money1 = money1 #private variable1!

    self.address1 = address1 #gloabal variable

  

  #get and set methods

  def getMoney(self):

    return self.__money1

  

  def setMoney(self, amount):

    self.__money1 = amount



  def increase(self):

    self.__money1 = self.__money1 + 500



person3 = BankAccount2("john", 400, "USA")

person4 = BankAccount2("messi", 600, "Canada")
#As we have seen, we cannot pull this object or variable.

# person3.__money



#But now we can access this data by using the get and set methods.

print("Get Method:", person3.getMoney())

person3.setMoney(300)

print("After Set Method:", person3.getMoney())
print("Set Method:", person4.getMoney())

person4.increase()

print("After Increase Method:", person4.getMoney())
#Inheritance



class Animals1(object):



  def __init__(self):

    print("animal is created")



  def walk(self):

    print("animal walk")

  

  def run(self):

    print("animal run")



class Bird(Animals1):

  

  def __init__(self):

    super().__init__()

    print("bird is created")

  

  def fly(self):

    print("bird can fly")



animal2 = Bird()

animal2.fly()
#Abstract Classes

#1. A superclass variable cannot be defined.

#2. A superclass defined method needs to be defined in child classes



from abc import ABC, abstractclassmethod



class Book(ABC):



  @abstractclassmethod

  def numberOfPages(self):

    pass  

  @abstractclassmethod

  def costBook(self):

    pass



class FunnyBook(Book):



  def __init__(self):

    print("Funny book")

  def numberOfPages(self):

    print("200 pages")

  def costBook(self):

    print("20.00 $")



book1 = FunnyBook()
#Overriding



class Fruits(object):



  def toString(self):

    print("Fruit created")

  

class Banana(Fruits):



  def toString(self):

    print("Banana!")



fruit1 = Fruits()

fruit1.toString()



fruit2 = Banana() # banana calls overriding method

fruit2.toString()
#Polymorphism



class Employee:



  def raisee(self):

    raisee_rate = 0.10

    result = 200 + 200 * raisee_rate

    print("Employee:", result)



class ComputerEngineer(Employee):



  def raisee(self):

    raisee_rate = 0.20

    result = 200 + 200 * raisee_rate

    print("Computer Engineer:", result)



class ElectricEngineer(Employee):



  def raisee(self):

    raisee_rate = 0.35

    result = 200 + 200 * raisee_rate

    print("ElectricEngineer:", result)



employee1 = Employee()

employee1.raisee()



computerengineer1 = ComputerEngineer()

computerengineer1.raisee()



electricengineer1 = ElectricEngineer()

electricengineer1.raisee()



employee_list = [computerengineer1, electricengineer1]

print(" ")



print("For each in employee_list")

for each in employee_list:

  each.raisee()
from abc import ABC, abstractclassmethod



class Shape(ABC):



  @abstractclassmethod

  def area(self): pass



  @abstractclassmethod

  def perimeter(self): pass



  def toString(self): pass



class Square(Shape):



  def __init__(self, edge):

    self.__edge = edge

  

  def area(self):

    result = self.__edge**2

    print("Square Area:", result)



  def perimeter(self):

    result = self.__edge*4

    print("Square Perimeter:", result)

  

  def toString(self):

    print("Square")

  

class Circle(Shape):



  PI = 3.14

  def __init__(self, radius):

    self.__radius = radius

  

  def area(self):

    result = Circle.PI*self.__radius**2

    print("Circle Area:", result)

  

  def perimeter(self):

    result = 2*Circle.PI*self.__radius

    print("Circle Perimeter:", result)

  

  def toString(self):

    print("Circle")



class Rectangle(Shape):

  

  def __init__(self, short_edge, long_edge):

    self.__short_edge = int(short_edge)

    self.__long_edge = int(long_edge)

  

  def area(self):

    result = (self.__short_edge * self.__long_edge)

    print("Rectangle Area:", result)



  def perimeter(self):

    result = (2 * (self.__short_edge + self.__long_edge))

    print("Rectangle Perimeter:", result)



  def toString(self):

    print("Rectangle")

  

square1 = Square(5)

square1.area()

square1.perimeter()

square1.toString()

print(" ")



circle1 = Circle(5)

circle1.area()

circle1.perimeter()

circle1.toString()

print(" ")



rectangle1 = Rectangle(4,5)

rectangle1.area()

rectangle1.perimeter()

rectangle1.toString()
#Create classes

import datetime



class VehicleRent:



  def __init__(self, stock):

    self.stock = stock

    self.now = 0

  

  def displayStock(self):

    print("{} vehicle available to rent".format(self.stock))

    return self.stock

  

  def rentHourly(self, n):

    if n <= 0:

      print("The value entered cannot be negative!")

      return None

    elif n > self.stock:

      print("Sorry, {} vehicle available to rent".format(self.stock))

      return None

    else:

      self.now = datetime.datetime.now()

      print("Rented a {} vehicle for hourly at {} hours".format(n,self.now.hour))



      self.stock -= n

      return self.now

    

  def rentDaily(self, n):

    if n <= 0:

      print("The value entered cannot be negative!")

      return None

    elif n > self.stock:

      print("Sorry, {} vehicle avaiable to rent".format(self.stock))

      return None

    else:

      self.now = datetime.datetime.now()

      print("Rented a {} vehicle for daily at {} hours".format(n,self.now.hour))



      self.stock -= n

      return self.now

  def returnVehicle(self, request, brand):

    car_h_price = 100

    car_d_price = car_h_price/8*24

    bike_h_price = 50

    bike_d_price = bike_h_price/8*24



    rentalTime, rentalBasis, numofVehicle = request

    rentalTime = self.now

    bill = 0



    if (brand == "car"):

      if rentalTime and rentalBasis and numofVehicle:

        self.stock += numofVehicle

        now = datetime.datetime.now()

        rentalPeriod = now - rentalTime



        if rentalBasis == 1: #hourly

          bill = rentalPeriod.seconds/3600*car_h_price*numofVehicle

        elif rentalBasis == 2: #daily

          bill = rentalPeriod.seconds/(3600*24)*car_d_price*numofVehicle

        

        if (2 <= numofVehicle):

          print("You have extra 20% discount!")

          bill = bill*0.8

        

        print("Thank you for returning your car")

        print("Price: $ {}".format(bill))

        return bill

    

    elif (brand == "bike"):

      if rentalTime and rentalBasis and numofVehicle:

        self.stock += numofVehicle

        now = datetime.datetime.now()

        rentalPeriod = now - rentalTime



      if (rentalBasis == 1): #Hourly

        bill = rentalPeriod.seconds/3600*bike_h_price*numofVehicle

      elif (rentalBasis == 2): #Daily

        bill = rentalPeriod.seconds/(3600*24)*bike_d_price*numofVehicle



      print("Thank you for returning your bike")

      print("Price: $ {}".format(bill))

      return bill

    

    else:

      print("You did not rent a car!")

      return None
class CarRent(VehicleRent):



  global discount_rate

  discount_rate = 15

  def __init__(self, stock):

    super().__init__(stock)

  

  def discount(self, b):

    bill = b - (b * discount.rate)/100

    return bill



class BikeRent(VehicleRent):



  def __init__(self, stock):

    super().__init__(stock)



class Customer:



  def __init__(self):

    self.bikes = 0

    self.cars = 0

    self.rentalBasis_b = 0

    self.rentalBasis_c = 0

    self.rentalTime_b = 0

    self.rentalTime_c = 0 

  

  def requestVehicle(self, brand):

    if brand == "bike":

      bikes = input("How many bikes would you like to rent?")

      

      try: 

        bikes = int(bikes)

      except ValueError:

        print("Number should be Number!")

        return -1

      

      if bikes < 1:

        print("Number of bikes should be greater than zero")

        return -1

      else:

        self.bikes = bikes

        return self.bikes



    elif brand == "car":

      cars = input("How many cars would you like to rent?")



      try: 

        cars = int(cars)

      except ValueError:

        print("Number should be Number!")

        return -1



      if cars < 1:

        print("Number of bikes should be greater than zero")

        return -1

      else:

        self.cars = cars

        return self.cars



    else:

      print("Request vehicle error")

      return None



  def returnVehicle(self, brand):

    if brand == "bike":

      if self.rentalBasis_b and self.rentalTime_b and self.bikes:

        return self.rentalBasis_b, self.rentalTime_b, self.bikes

      else:

        return 0,0,0

    

    elif brand == "car":

        if self.rentalBasis_c and self.rentalTime_c and self.cars:

          return self.rentalBasis_c, self.rentalTime_c, self.cars

        else:

          return 0,0,0

    else:

      print("Return vehicle error")
bike100 = BikeRent(100)

car10 = CarRent(10)

customer11 = Customer()
main_menu = True



while True:



  if main_menu:

    print("""

    **** Vehicle Rental Shop ****

    A. Bike Menu

    B. Car Menu

    Q. Exit

    """)

    main_menu = False



    choice = input("Enter choice: ")



  if choice == "A" or choice == "a":

    print("""

    **** Bike Menu ****

    1. Display Available Bikes

    2. Request a bike on hourly basis $50

    3. Request a bike on daily basis $160

    4. Return a Bike

    5. Main Menu

    6. Exit

    """)



    choice = input("Enter choice: ")



    try:

      choice = int(choice)

    except ValueError:

      print("It is not integer")

      continue

    

    if choice == 1:

      bike100.displayStock()

      choice = "A"

    elif choice == 2:

      customer11.rentalTime_b = bike100.rentHourly(customer11.requestVehicle("bike"))

      main_menu = True

      print("----------------")

    elif choice == 3:

      customer11.rentalTime_b = bike100.rentDaily(customer11.requestVehicle("bike"))

      customer11.rentalBasis_b = 2

      main_menu = True

      print("----------------")

    elif choice == 4:

      customer11.bill = bike100.returnVehicle(customer11.returnVehicle("bike"),"bike")

      customer11.rentalBasis_b, customer11.rentalTime_b, customer11.bikes = 0,0,0

      main_menu = True

      print("----------------")

    elif choice == 5:

      main_menu = True

    elif choice == 6:

      break

    else:

      print("Invalid input! Please enter a number between 1-6")

      main_menu = True



  elif choice == "B" or choice == "b":

    print("""

    **** Car Menu ****

    1. Display Available Cars

    2. Request a car on hourly basis $50

    3. Request a car on daily basis $160

    4. Return a Car

    5. Main Menu

    6. Exit

    """)



    choice = input("Enter a choice: ")



    try:

      choice = int(choice)

    except ValueError:

      print("It is not integer")

      continue

    

    if choice == 1:

      car10.displayStock()

      choice = "B"

    elif choice == 2:

      customer11.rentalTime_c = car10.rentHourly(customer11.requestVehicle("car"))

      customer11.rentalBasis_c = 1

      main_menu = True

      print("-------------------")

    elif choice == 3:

      customer11.rentalTime_c = car10.rentDaily(customer11.requestVehicle("car"))

      customer11.rentalBasis_c = 2

      main_menu = True

      print("-------------------")

    elif choice == 4:

      customer11.bill = car10.returnVehicle(customer11.returnVehicle("car"),"car")

      customer11.rentalBasis_c, customer11.rentalTime_c, customer11.cars = 0,0,0

      main_menu = True

      print("-------------------")

    elif choice == 5:

      main_menu = True

    elif choice == 6:

      break

    else:

      print("Invalid input! Please enter a number between 1-6")

      main_menu = True



  elif choice == "Q" or choice == "q":

    break

  

  else:

    print("Invalid input! Please enter A-B-Q")

    main_menu = True

  print("Thank you for using Rental Vehicle Shop!")
def addition(value1, value2):

  return value1 + value2



def subtract(value1, value2):

  return value1 - value2



def division(value1, value2):

  return value1/value2



def multiply(value1, value2):

  return value1 * value2



while True:



  try:

    value1 = float(input("Enter a value1: "))

    value2 = float(input("Enter a value2: "))

    operator = input("""

    What is the procedure to do(addition, subtract, division, multiply): 

    Note: Press (Q) to sign out!:

    """)



    if operator == "addition":

      addition_o = addition(value1, value2)

      print("Addition: {}".format(addition_o))

      break

    

    elif operator == "subtract":

      subtract_o = subtract(value1, value2)

      print("Subtract: {}".format(subtract_o))

      break

    

    elif operator == "divison":

      division_o = division(value1, value2)

      print("Divison: {}".format(division_o))

      break



    elif operator == "multiply":

      multiply_o = multiply(value1, value2)

      print("Multiply: {}".format(multiply_o))

      break

    

    elif operator == "q":

      break

     

    else:

      print("An incorrect value was entered.")

  except ValueError:

    print("An incorrect value was entered.")
import time



rentalTime4 = time.ctime()

print(rentalTime4)

rentalTime5 = time.ctime()

print(rentalTime5)
import datetime

rentalTime6 = datetime.datetime.now()

print(rentalTime6)
rentalTime7 = datetime.datetime.now()

print(rentalTime7)



d = rentalTime7 - rentalTime6

print(d.seconds/3600*24*100)
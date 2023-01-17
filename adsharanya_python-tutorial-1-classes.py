#This was created as a refresher for myself. 
#Follow comments to understand class object, methods, instance attributes, methods and subclasses.

#This was created by following these youtube videos:
#    https://www.youtube.com/watch?v=ZDa-Z5JzLYM
#    https://www.youtube.com/watch?v=BJ-VvGyQxho
#    https://www.youtube.com/watch?v=rq8cL2XMM5M
#    https://www.youtube.com/watch?v=RSl87lqOXDE
#    Thanks to Corey Schafer

class Person:    #this is the name of the python class
    
    #these are class_variables. These will be common for all instances of this class.
    class_height = 100
    class_weight = 20
    num_of_persons = 0
    
    #this is the initi function for the instance variable
    def __init__(self, name, height, weight): #all instance method's 1st input is the instance.
                                              #by convention this is self
        self.name = name
        self.height = height
        self.weight = weight
        Person.num_of_persons += 1    #you can change a class attribute this way.
        
    #this is an example of instance method
    def change_height(self,height_from_user): #it has self as 1st input (which is required). the 2nd variable is 
                                            #what will be passed by user when this methos will be called.
        self.height = height_from_user
        
    #below is an example of class method    
    @classmethod
    def change_height_of_class(cls,send_height):
        cls.class_height = send_height
        
    #below is an example of using class method as alternative constructor
    #say the input is in a different format then you would call the below method to 
    #create an instance
    @classmethod #this is called a decorator
    def from_string(cls,input_string):
        name, height, weight = input_string.split('-')
        return cls(name, height, weight)
    
    
    #below is an example of static method
    #static methods dont call self(i.e., instance) or cls(i.e., class) 
    #python has method called weekday, if day = Monday, it'll return 0.
    #if Sunday it'll return 6, etc.
    #the below static method was created to check if a given day is Saturday or Sunday
    
    @staticmethod 
    def check_day(day):
        if day.weekday == 5 or day.weekday == 6:
            return True
        return False
    
    #below is an example of inheritance. The class Employed_Person (is a subclass of Person) calls 
    #the class Person. When an instance of Employed_Person is created, it'll create
    #all attributes and methods of class Person      
class Employed_Person(Person):
    pass

   #below I'm going to creat a bit more complicated subclass
class Employed_Person_complex(Person):
    class_height = 200 #this gets precedace over the class_height attribute of class Person
    
    def __init__(self,name,height,weight,job):
        super().__init__(name,height,weight)     #this lets you use name, height and weight from class Person
        self.job = job                           #this lets you set the additional attribute job for subclass Employed_Person_Complex
   
   #below I'm creating another subclass called Supervisors. These supervisors will have employed persons reporting to them.
#We want to keep track of the persons reporting to the supervisors

class Supervisor(Person):
    
    def __init__(self, name, height, weight, reporting_employees=None):   #the init function has the new attribute reporting_employees
        super().__init__(name, height, weight)
        if reporting_employees is None:
            self.reporting_employees = []
        else:
            self.reporting_employees = reporting_employees
            
    def add_employee(self, new_employee):                         #add a new employee
        if new_employee not in self.reporting_employees:
            self.reporting_employees.append(new_employee)
            
    def del_employee(self, del_employee):                      #del employee
        self.reporting_employees.remove(del_employee)
        
    def show_employees(self):                                  #display all employees
        for emp in self.reporting_employees:
            print('------->', emp.name)
            print('----employee job is ', emp.job)
        
#Initilizing instance variables
print(Person.num_of_persons)   
person_1 = Person('Joe','150','30')
person_2 = Person('Jill','200','36')
print(Person.num_of_persons)
#Now changing a few things..
person_1.change_height(110)
print(person_1.height)
print(person_1.class_height)
Person.change_height_of_class(120)
print(Person.class_height)
#Input string as input (call class method)
person_3 = Person.from_string('Jack-220-34')
import datetime
my_time = datetime.date(2016,10,10)
Person.check_day(my_time)   #checking if given date is Saturday or Sunday

person_1.__dict__  #do dict to check attributes associted with an instance or class
Person.__dict__
print(help(Employed_Person)) #use this help function to see the method order resolution for inherited classes.
emp_person_1 = Employed_Person_complex('Mary', 150, 39, 'pilot')
emp_person_2 = Employed_Person_complex('Julie', 150, 39, 'cabdriver')
emp_person_3 = Employed_Person_complex('Sandra', 150, 39, 'seacaptain')
sup_person_1 = Supervisor('Sheryl', 170, 40, [])
sup_person_1.add_employee(emp_person_1)
sup_person_1.add_employee(emp_person_2)
sup_person_1.add_employee(emp_person_3)
sup_person_1.show_employees()
sup_person_1.del_employee(emp_person_2)
sup_person_1.show_employees()
#check if instance is of a class
print(isinstance(sup_person_1,Person))
print(isinstance(sup_person_1,Supervisor))
print(isinstance(sup_person_1,Employed_Person_complex))


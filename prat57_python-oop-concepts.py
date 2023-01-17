# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
class Employee:
    
    def __init__(self,first,last,salary):
        self.first = first
        self.last = last
        self.salary = salary
        self.email = first + '.' + last + '.' + '@company.com'
    
    def fullname(self):
        return '{} {}'.format(self.first,self.last)
    
emp1 = Employee('Pratik','Shinde',60000)
emp2 = Employee('Adam','West',50000)

emp1.fullname()
print(Employee.fullname(emp1))
#Both of these commands give same results
    
    
class Employee:
    
    raise_amount = 1.04
    no_of_employees = 0
    def __init__(self,first,last,pay):
        self.first = first
        self.last = last
        self.pay = pay
        self.email = first + '.' + last + '.' + '@company.com'
        
        Employee.no_of_employees +=1 #Counter
        
    def fullname(self):
        return '{} {}'.format(self.first,self.last)
    
    def apply_raise(self):
        self.pay = int(self.pay * Employee.raise_amount)
        # Instead of hardcoding the raise amount, we can declare it a class variable, so need to change only at one place without changing it everywhere
    
print(Employee.no_of_employees)    
emp1 = Employee('Pratik','Shinde',60000)
emp2 = Employee('Priti','Trivedi',90000)
emp1.apply_raise()
print(Employee.no_of_employees)
# Notice the no of employees before and after creating the instances

print(emp1.__dict__)
print(emp2.__dict__)
print(Employee.__dict__)

# Here raise_amount and no_of_employees are Class variables that's why we can't see them in the instance dictionaries. We can see them in class dictionary
# Regular methods take class instances as their argument(e.g. self)
# We can turn any regular method into a class method by adding decorator @classmethod.. These methods take class as their first instance.
# Static methods don't take instance or the class as their first argument. So we can work with anything.

class Employee:
    
    raise_amount = 1.04
    no_of_employees = 0
    def __init__(self,first,last,pay):
        self.first = first
        self.last = last
        self.pay = pay
        self.email = first + '.' + last + '.' + '@company.com'
        
        Employee.no_of_employees +=1 #Counter
        
    def fullname(self):
        return '{} {}'.format(self.first,self.last)
    
    def apply_raise(self):
        self.pay = int(self.pay * Employee.raise_amount)
       
    @classmethod
    def set_raise_amt(cls, amount):
        cls.raise_amt = amount
        
    @classmethod
    def from_string(cls, emp_str):
        first, last, pay = emp_str.split('-')
        return Employee(first, last, pay)
        #return cls(first, last, pay)  Same as above
        
    @staticmethod
    def is_workday(day):
        if day.weekday()==5 or day.weekday()==6:
            return False
        return True
        
    
  
emp1 = Employee('Pratik','Shinde',60000)
emp2 = Employee('Adam','West',50000)

emp_str1 = 'John-Doe-30000'
emp_str2 = 'Virat-Kohli-40000'
emp_str3 = 'Optimus-Prime-50000'

new_emp1 = Employee.from_string(emp_str1) # Have to call a classmethod by class name
new_emp2 = Employee.from_string(emp_str2)
new_emp3 = Employee.from_string(emp_str3)

print(new_emp1.first)

Employee.set_raise_amt(1.09) #Another way of setting raising amount for every instance of the class at once by using a class method
print(emp1.raise_amt)
print(emp2.raise_amt)

import datetime
my_date =  datetime.date(2020, 8, 13)

print(Employee.is_workday(my_date)) #Static methods are also called with class names

class Employee:

    raise_amt = 1.04

    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.email = first + '.' + last + '@email.com'
        self.pay = pay

    def fullname(self):
        return '{} {}'.format(self.first, self.last)

    def apply_raise(self):
        self.pay = int(self.pay * self.raise_amt)


class Developer(Employee):
    raise_amt = 1.10

    def __init__(self, first, last, pay, prog_lang):
        super().__init__(first, last, pay)
        self.prog_lang = prog_lang


class Manager(Employee):

    def __init__(self, first, last, pay, employees=None):
        super().__init__(first, last, pay)
        if employees is None:
            self.employees = []
        else:
            self.employees = employees

    def add_emp(self, emp):
        if emp not in self.employees:
            self.employees.append(emp)

    def remove_emp(self, emp):
        if emp in self.employees:
            self.employees.remove(emp)

    def print_emps(self):
        for emp in self.employees:
            print('-->', emp.fullname())


dev_1 = Developer('Corey', 'Schafer', 50000, 'Python')
dev_2 = Developer('Test', 'Employee', 60000, 'Java')

mgr_1 = Manager('Sue', 'Smith', 90000, [dev_1])

print(mgr_1.email)

mgr_1.add_emp(dev_2)
mgr_1.remove_emp(dev_2)

mgr_1.print_emps()
class Employee:

    raise_amt = 1.04

    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.email = first + '.' + last + '@email.com'
        self.pay = pay

    def fullname(self):
        return '{} {}'.format(self.first, self.last) #using placeholders

    def apply_raise(self):
        self.pay = int(self.pay * self.raise_amt)

    def __repr__(self):
        return "Employee('{}', '{}', {})".format(self.first, self.last, self.pay)

    def __str__(self):
        return '{} - {}'.format(self.fullname(), self.email)

    def __add__(self, other):
        return self.pay + other.pay

    def __len__(self):
        return len(self.fullname())


emp_1 = Employee('Corey', 'Schafer', 50000)
emp_2 = Employee('Test', 'Employee', 60000)

# print(emp_1 + emp_2)

print(len(emp_1))
class Employee:

    def __init__(self, first, last):
        self.first = first
        self.last = last

    @property        # Getter
    def email(self):
        return '{}.{}@email.com'.format(self.first, self.last)

    @property        # Getter
    def fullname(self):
        return '{} {}'.format(self.first, self.last)
    
    @fullname.setter
    def fullname(self, name):
        first, last = name.split(' ')
        self.first = first
        self.last = last
    
    @fullname.deleter
    def fullname(self):
        print('Delete Name!')
        self.first = None
        self.last = None


emp_1 = Employee('John', 'Smith')
emp_1.fullname = "Corey Schafer"

print(emp_1.first)
print(emp_1.email)
print(emp_1.fullname)

del emp_1.fullname

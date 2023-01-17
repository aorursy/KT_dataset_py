# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
class MyClass:

  x = 5

p1 = MyClass()

print(p1.x)
class Person1:

  def __init__(self, name, age):

    self.name = name

    self.age = age



p1 = Person1("John", 36)



print(p1.name)

print(p1.age)
class Person2:

  def __init__(self, name, age):

    self.name = name

    self.age = age



  def myfunc(self):

    print("Hello my name is " + self.name)

    print("Hello my age is  ", self.age)



p1 = Person2("John", 36)

p1.myfunc()
# Set and Print Age

p1.age = 40

p1.myfunc()
del p1.age
p1.myfunc()
class Student(Person2):

  pass
x = Student("Mike", "Olsen")

x.myfunc()
class Student(Person2):

  def __init__(self, fname, lname):

     return
class Student(Person2):

  def __init__(self, fname, lname):

    Person.__init__(self, fname, lname)
class Student(Person2):

  def __init__(self, fname, lname):

    super().__init__(fname, lname)
class Student(Person2):

  def __init__(self, fname, lname, year):

    super().__init__(fname, lname)

    self.graduationyear = year



x = Student("Mike", "Olsen", 2019)
class Student(Person2):

  def __init__(self, fname, lname, year):

    super().__init__(fname, lname)

    self.graduationyear = year



  def welcome(self):

    print("Welcome", self.firstname, self.lastname, "to the class of", self.graduationyear)
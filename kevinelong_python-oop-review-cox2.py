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
# ##
# VARIABLES AND DATA TYPES
###

# Variable: A named reference to a value
quantity = 144

# Data Type: A Specific kind of variable (e.g. Integer, String, List)

# Number: numbers come in two flavors
# Integer: whole numbers (e.g. 1,2,3)
i = 12
# Float: A floating-point/decimal number can hold fractional values (e.g. 1.0, 0.25 13.79)
f = 33.33
half = 1.0 / 2.0

# String: An ordered sequence of characters
student_name = "Alice Jones"

# List: A list is an ordered collection of values that can be referenced by position in the list [0,1,2,3,...]
# 0 is the index of the first item, 1 is the second item 2 is the third item etc
# -1 is the last item -2 is the second to the last item
people = ["Bob", "Carol", "Ted", "Alice"]
first_person = people[0]
last_person = people[-1]

quarterly_sales_for_year = [100, 75, 50, 200]
q3 = quarterly_sales_for_year[2]

#Dictionary: a dictionary is an unordered collection of values that can be accessed by a name known as a key like a phone book or library
phone_book = {"Bob": "555-555-2222", "Carol": "555-555-3333", "Ted": "555-555-4444", "Alice": "555-555-1111"}
alice_number = phone_book["Alice"]

#Boolean: A Boolean is a binary logical value. e.g. True or False
is_a_student = True
is_cheating = False


###
# Classes and Modules
###
class Person(object):
    
    #Constructor Method
    def __init__(self, first_name, last_name):
        self.first_name = first_name
        self.last_name = last_name

    #Another Method
    def full_name(self):
        return self.first_name + " " + self.last_name


#Encapsulation

class Group(list):
    def __init__(self, people):
        self.people = people

    def show(self):
        for person in self.people:
            print(person.full_name())


p = Person("Kevin", "Long")

print(p.first_name)

g = Group([
    p,
    Person("Ashley", "Ford"),
    Person("Bob", "Dobbs")
])

# data = list(1,1,1)
# data = [1,1,1]
# data = {}
# data dict(1=>"A", 2=>"B")

k = g.people[0]
print(k.first_name)

g.show()

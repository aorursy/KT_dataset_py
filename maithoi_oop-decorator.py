# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Property Decorator-Getter, Setter and Deleter

'''

Property Decorator allows us define a method, but we can acess it like  an attribute







'''

class Student:

    def __init__(self, first, last):

        self.first = first

        self.last = last

        self.email = first + '.' + last + '@XPro.com'



    def full_name(self):

        return '{} {}'.format(self.first, self.last)

    

student_1 = Student('Mai', 'Thoi')

student_1.first = 'Mai V.'





print('student_1.first:', student_1.first)

print('student_1.email:', student_1.email)

print('student_1.full_name():', student_1.full_name())

    
#Property Decorator-Getter, Setter and Deleter



class Student:

    #define __init__() method

    def __init__(self, first, last):

        #set attributes for the Student class

        self.first = first

        self.last = last





    #define email() method and full_name() method in the same way

    def email(self):

        return '{}{}@Xpro.com'.format(self.first, self.last)



    def full_name(self):

        return '{} {}'.format(self.first, self.last)

    

student_1 = Student('Mai', 'Thoi')

student_1.first = 'Mai V.'

 



print('student_1.first:', student_1.first)

#access the method of class

print('student_1.email():', student_1.email())

print('student_1.full_name():', student_1.full_name())

    
#Property Decorator-Getter, Setter and Deleter



class Student:

    #define __init__() method

    def __init__(self, first, last):

        #set attributes for the Student class

        self.first = first

        self.last = last





    #define email() method and full_name() method in the same way

    @property

    def email(self):

        return '{}{}@Xpro.com'.format(self.first, self.last)

    @property

    def full_name(self):

        return '{} {}'.format(self.first, self.last)

    

    #define a fullname setter

    @full_name.setter

    def full_name(self, name):

        first, last = name.split()

        self.first = first

        self. last = last

                

    #define a deleter to clean the full_name

    @full_name.deleter

    def full_name(self):

        print('\nempty the full_name')

        self.first = None

        self.last = None

    

student_1 = Student('Mai', 'Thoi')

student_1.first = 'Mai V.'

#define fullname with a defined setter decorator

student_1.full_name = 'Nhu Y'





print('student_1.first:', student_1.first)

#define  email() and full_name() method, but now we access them like an attribute

print('student_1.email:', student_1.email)

print('student_1.full_name:', student_1.full_name)



del student_1.full_name

     
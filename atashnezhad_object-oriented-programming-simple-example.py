import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
class OOP_test1:

        

    def __init__(self, name, constant_1, constant_2):

        

        self.name = name

        self.cons_1 = constant_1

        self.cons_2 = constant_2
case_1 = OOP_test1('name_as_you_wish', 0.5, 1)
print(case_1)
print('name is = ',case_1.name)

print('cosntant_1 value is =', case_1.cons_1)

print('cosntant_2 value is =', case_1.cons_2)
class OOP_test1:

        

    def __init__(self, name, constant_1, constant_2):

        

        self.name = name

        self.cons_1 = constant_1

        self.cons_2 = constant_2

        

        self.new_attribute = self.cons_1 + self.cons_2
case_1 = OOP_test1('name_as_you_wish', 0.5, 1)
print(case_1.new_attribute)
class OOP_test1:

        

    def __init__(self, name, constant_1, constant_2):

        

        self.name = name

        self.cons_1 = constant_1

        self.cons_2 = constant_2

        

        self.new_attribute = self.cons_1 + self.cons_2

        

    def Method_name_1(self, x1, x2):

        '''You may name the method as you wish.

        this method takes two input parameters, two direct introduced values (x1, x2) and

        it uses two already introduced values (cons1 and cons2)

        and returns a value'''

        return (x1+x2)/(self.cons_1+self.cons_2)
case_1 = OOP_test1('name_as_you_wish', 0.5, 1)
case_1.Method_name_1(1.2,5.4)
class OOP_test1:

        

    def __init__(self, name, constant_1, constant_2):

        

        self.name = name

        self.cons_1 = constant_1

        self.cons_2 = constant_2

        

        self.new_attribute = self.cons_1 + self.cons_2

        

    def Method_name_1(self, x1, x2):

        '''You may name the method as you wish.

        this method takes two input parameters, two direct introduced values (x1, x2) and

        it uses two already introduced values (cons1 and cons2)

        and returns a value'''

        return (x1+x2)/(self.cons_1+self.cons_2)

    

    def Method_name_2(self, x1, x2, x3, x4):

        '''You may name the method as you wish.

        this method takes four input parameters.'''

        

        OutPut_Method_name_1 = self.Method_name_1(x1, x2)

        

        OutPut_Method_name_2 = OutPut_Method_name_1 + x3*x4

        

        return OutPut_Method_name_2
case_1 = OOP_test1('name_as_you_wish', 0.5, 1)

case_1.Method_name_2(1.2, 5.4, 1, 10)
case_2 = OOP_test1('name_as_you_wish', 6.5, 4)

case_2.Method_name_2(1.2, 5.4, 1, 10)
print('case_1 attributes =', case_1.__dict__)

print('case_2 attributes =', case_2.__dict__)
class OOP_test1:

        

    def __init__(self, name, constant_1, constant_2):

        

        self.name = name

        self.cons_1 = constant_1

        self.cons_2 = constant_2

        

        self.new_attribute = self.cons_1 + self.cons_2

        

    def Method_name_1(self, x1, x2):

        '''You may name the method as you wish.

        this method takes two input parameters, two direct introduced values (x1, x2) and

        it uses two already introduced values (cons1 and cons2)

        and returns a value'''

        return (x1+x2)/(self.cons_1+self.cons_2)

    

    def Method_name_2(self, x1, x2, x3, x4):

        '''You may name the method as you wish.

        this method takes four input parameters.'''

        

        OutPut_Method_name_1 = self.Method_name_1(x1, x2)

        

        OutPut_Method_name_2 = OutPut_Method_name_1 + x3*x4

        

        self.OutPut_Method_name_2 = OutPut_Method_name_2

        

        return OutPut_Method_name_2
case_1 = OOP_test1('name_as_you_wish', 0.5, 1)

case_1.Method_name_2(1.2, 5.4, 1, 10)

case_2 = OOP_test1('name_as_you_wish', 6.5, 4)

case_2.Method_name_2(1.2, 5.4, 1, 10)

print('case_1 attributes =', case_1.__dict__)

print('case_2 attributes =', case_2.__dict__)
class OOP_test1:

        

    def __init__(self, name, constant_1, constant_2):

        

        self.name = name

        self.cons_1 = constant_1

        self.cons_2 = constant_2

        

        self.new_attribute = self.cons_1 + self.cons_2

        

    def Method_name_1(self, x1, x2):

        '''You may name the method as you wish.

        this method takes two input parameters, two direct introduced values (x1, x2) and

        it uses two already introduced values (cons1 and cons2)

        and returns a value'''

        return (x1+x2)/(self.cons_1+self.cons_2)

    

    def Method_name_2(self, x1, x2, x3, x4):

        '''You may name the method as you wish.

        this method takes four input parameters.'''

        

        OutPut_Method_name_1 = self.Method_name_1(x1, x2)

        

        OutPut_Method_name_2 = OutPut_Method_name_1 + x3*x4

        

        self.OutPut_Method_name_2 = OutPut_Method_name_2

        

        return OutPut_Method_name_2

    

    def Plot(self, x1, x2):

        

        

        self.x_values = [self.cons_1, self.cons_2]

        self.y_values = [x1, x2]

        

        plt.plot(self.x_values,self.y_values,'b^', markersize=4)

        plt.xlabel('X')

        plt.ylabel('Y')

        

        

        
case_1 = OOP_test1('name_as_you_wish', 0.5, 1)

case_2 = OOP_test1('name_as_you_wish', 6.5, 4)
case_1.Plot(2,5)
case_2.Plot(2,2)
class OOP_test1:

    

    

    instances = [] # This line was added.

        

    def __init__(self, name, constant_1, constant_2):

        

        self.__class__.instances.append(self) # This line was added.

        

        self.name = name

        self.cons_1 = constant_1

        self.cons_2 = constant_2

        

        self.new_attribute = self.cons_1 + self.cons_2

        

    def Method_name_1(self, x1, x2):

        '''You may name the method as you wish.

        this method takes two input parameters, two direct introduced values (x1, x2) and

        it uses two already introduced values (cons1 and cons2)

        and returns a value'''

        return (x1+x2)/(self.cons_1+self.cons_2)

    

    def Method_name_2(self, x1, x2, x3, x4):

        '''You may name the method as you wish.

        this method takes four input parameters.'''

        

        OutPut_Method_name_1 = self.Method_name_1(x1, x2)

        

        OutPut_Method_name_2 = OutPut_Method_name_1 + x3*x4

        

        self.OutPut_Method_name_2 = OutPut_Method_name_2

        

        return OutPut_Method_name_2

    

    def Plot(self, x1, x2):

        

        

        self.x_values = [self.cons_1, self.cons_2]

        self.y_values = [x1, x2]

        

        plt.plot(self.x_values,self.y_values,'b^', markersize=4)

        plt.xlabel('X')

        plt.ylabel('Y')

    

    

    @classmethod

    def Plot_whole_instances(cls):

        i = 1

        for instance in cls.instances:

            

            plt.subplot(1, 2, i)

            plt.plot(instance.cons_1,instance.cons_2,'bo', markersize=4)

            i += 1
case_1 = OOP_test1('name_as_you_wish', 0.5, 1)

case_2 = OOP_test1('name_as_you_wish', 6.5, 4)
OOP_test1.Plot_whole_instances()
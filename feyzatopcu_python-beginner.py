

# Define variables

entry = "Software Engineer"

name = "Feyza"

age = 23

city = "Ankara"

Python_Beginner = (1 == True)



# print entry variable value

print("entry : ",entry)
# print Python_Beginner variable value

print("Python_beginner: : ",Python_Beginner)
#Assignment of more than one variable in a single line

name, age, city = "Feyza",23,"Ankara"
print(type(entry),type(name),type(age),type(city),type(Python_Beginner))
cities = ['barcelona', 'spain', 'germany', 'england']
cities[2]
cities[1]
cities[-1]
cities[::]
cities[1::1]
cities[1::3]
cities[:4]
cities[::-1]
cities[0:3] = ['france ','azerbaijan']
cities
cities.append('russia ')

cities
cities.append('spain')

cities
cities.append('germany')

cities
cities.remove('germany')

cities
cities = cities + ['china ']

cities
#list methods

cities.reverse()

cities
cities.sort()

cities
#Reference type example (method 1)

plaque = [67,6,45,34,35]
# wrong list copy method

copy_plaque= plaque
#is the change made only in the copy_plaque object?

copy_plaque[1] = 20
#Indices number 1 as a result of the change:

print(plaque)

print(copy_plaque)

#other method for copying list (method 2)

plaque = [67,6,45,34,35]

#legal copy methods

copy_plaque = plaque[::] #copy_plaque = list(plaque)
# Does the 0-indexed element of both objects change?

copy_plaque[1] = 23

print("plaque: ",plaque)

print("copy_plaque: ",copy_plaque)
#dictionary usage

notes = {

    "0001-Zeynep": 88,

    "0002-Emin": 99,

    "0003-Reyhan" : 92, 

    "0004-Şüheda" : 85 

}



#access to dictionary element

notes["0001-Zeynep"] #88
# add new element to dictionary

notes ["0005-Hilmi"] = 79

notes
# delete element in dictionary

del (notes ["0005-Hilmi"])

notes
("0004-Şüheda" in notes) #True
# dictionary methods
notes.keys()
notes.values() #dict_values([88,99,92,85])
notes.items()
#function definition

def bubble_Sequence(array):

    element_number = len(array)

    # Return all elements

    for i in range(element_number):

        for j in range(0, element_number - i - 1):

            #Substitution

            if array[j] > array[j+1] :

                array[j], array[j+1] = array[j+1], array[j]
#function usage

numbers = [3, 88, 99, 148, 5, 8, 11, 214, 2, 1]
bubble_Sequence(numbers)

for i in range(len(numbers)):

    print('%d' %numbers[i]) 
class BaseClassifier(object):

    def __init__(self):

        pass

    def get_name(self):

        raise NotImplementedError()

    def fit(self, x, y):

        raise NotImplementedError()

    def predict(self, x):

        raise NotImplementedError()
class BaseTree(BaseClassifier):

    def __init__(self, maks_derinlik):

        super().__init__()

        self.height = height



    def get_name(self):

        return "Base Tree";

 

    def fit(self, x, y):

        self.X = x

        self.Y = y

 

    def predict(self, x):

        y = np.zeros((x.shape[0],))

        for i in range(x.shape[0]):

            y[i] = self.__predict_single(x[i, :])

        return y
import numpy as np



#Creating an array

d1 = np.array([5.5,9,10])

d2 = np.array([(3.5,8,11), (4,7,9), (2,2,1.1)], dtype=float)



#Difference 1. Methods

d3 = d2 - d1

print ("Difference 1 / d3 ->", d3)



#Difference Method 2

d3 = np.subtract(d1, d2) 

print("Difference 2 / d3 --> ", d3)



#Adding # d1 and d2 and overwriting d1

d1 = d1 + d2

print ("Total d1 ->", d1)

d1
#Find indexes of elements whose value is greater than 3

result = d1> 9

print (result)
#Print elements on screen using found indices

print ("Elements greater than 3 ->", d1 [result])
# Product of two matrices

import numpy as np

d4 = np.dot(d1,d2)

print ("Multiplication d4:", d4)



#Removing the 1st column from the matrix

d4 = np.delete (d4,0,1)

print ("Subtraction d4:", d4)



#Creating a # 2x5 zero matrix

Zero_matrice = np.zeros([2,5])
#Find the smallest element in the array

print ("d4 min:", np.min (d4))



#Finding the largest element in the array

print ("d4 max:", np.max (d4))



#Average of the index

print ("d4 mean:", d4.mean ())



#Find the sum of the index

print ("d4 total:", d4.sum ())



# Square root

print ("d4 square root ->", np.sqrt (d4))



#Calculating the logarithm of the array

print ("d4 logarithm ->", np.log (d4))



#Transposition

print ("d4 transpos:", np.transpose (d4))
list1 = [10, 92, 83, 94, 15, 36]

list2 = [i/2 for i in list1]

print(list2) 
x, y = 5, 5

if (x > y):

    print("x > y")

elif (y > x):

    print("y > x")

else:

    print("y = x")
#while cycle

condition, j = True, 0

while (condition):

    print(j)

    j += 1

    condition = (j != 5)





#for cycle

for i in range(0, 5):

    print(i)
 #lamp function Function definitions containing

def fnc3(n):

  return lambda x : x ** n



fnc_square = fnc3(2)#Dynamic squaring function created

fnc_cube = fnc3(3) # Creating dynamic cube import function





print(fnc_square(3))

print(fnc_cube(3))
import pandas as pd

data = [

        ['D1', 'Sunny','Hot', 'High', 'Weak', 'No'],

        ['D2', 'Sunny','Hot', 'High', 'Strong', 'No'],

        ['D3', 'Overcast','Hot', 'High', 'Weak', 'Yes'],

        ['D4', 'Rain','Mild', 'High', 'Weak', 'Yes'],

        ['D5', 'Rain','Cool', 'Normal', 'Weak', 'Yes'],

        ['D6', 'Rain','Cool', 'Normal', 'Strong', 'No'],

        ['D7', 'Overcast','Cool', 'Normal', 'Strong', 'Yes'],

        ['D8', 'Sunny','Mild', 'High', 'Weak', 'Yes'],

        ['D9', 'Sunny','Cool', 'Normal', 'Weak', 'No'],

        ['D10', 'Rain','Mild', 'Normal', 'Weak', 'Yes'],

        ['D11', 'Sunny','Mild', 'Normal', 'Strong', 'Yes'],

        ['D12', 'Overcast','Mild', 'High', 'Strong', 'No'],

        ['D13', 'Overcast','Hot', 'Normal', 'Weak', 'Yes'],

        ['D14', 'Rain','Mild', 'High', 'Strong', 'No'],

       ]

df = pd.DataFrame(data,columns=['day', 'outlook', 'temp', 'humidity', 'windy', 'play'])

df
df.info()
df.max()
df.min()
df.describe()
df.shape
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder() 

df['outlook'] = lb.fit_transform(df['outlook']) 

df['temp'] = lb.fit_transform(df['temp'] ) 

df['humidity'] = lb.fit_transform(df['humidity'] ) 

df['windy'] = lb.fit_transform(df['windy'] )   

df['play'] = lb.fit_transform(df['play'] ) 
df
df.describe()
X = df.iloc[:,1:3] 

Y = df.iloc[:,3]
X
Y
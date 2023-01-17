import random

print ("A random number between 0 and 1 is : ", end="") 

print (random.random()) 
print("Random number intializing a seed value : ")

random.seed(10)

print(random.random())



random.seed(10)

print(random.random())



print("Random number without intializing a seed value : ")

print(random.random())
rn=random.randint(1,10)

print("The random integer between the specified range is : ",rn)
print ("A random number from range is : ",end="") 

print (random.randrange(1,10))
print ("A random number from list is : ",end="") 

print (random.choice([1, 4, 8, 10, 3])) 

# String manipulation:

x = "WELCOME"

print("Random character from the given string : ",random.choice(x))
mylist = ["apple", "banana", "cherry"]

print("The randomly selected items are:")

print(random.sample(mylist, k=2))



string = "PythonProgramming"

print("With string:", random.sample(string, 4))



# Prints list of random items of length 4 from the given tuple. 

tuple1 = ("ankit", "geeks", "computer", "science", 

                   "portal", "scientist", "btech") 

print("With tuple:", random.sample(tuple1, 4)) 

  

#Prints list of random items of length 3 from the given set. 

set1 = {"a", "b", "c", "d", "e"} 

print("With set:", random.sample(set1, 3))
li = [1, 4, 5, 10, 2]

print("The list after shuffling :")

random.shuffle(li)

print(li)



mylist = ["apple", "banana", "cherry"]

random.shuffle(mylist)

print("mylist after shuffling : ")

print(mylist)



from random import shuffle

x = [i for i in range(10)]

random.shuffle(x)

print("x :",x)
print(random.uniform(20, 60))
import numpy as np 

   

# 1D Array 

array = np.random.randn(5) 

print("1D Array filled with random values : \n", array)

array = np.random.randn(3, 4) 

print("2D Array filled with random values : \n", array);

array = np.random.randn(2, 2 ,2) 

print("3D Array filled with random values : \n", array); 
from scipy.stats import rv_discrete

values = [10, 20, 30]

probabilities = [0.2, 0.5, 0.3]

distrib = rv_discrete(values=(values, probabilities))

distrib.rvs(size=10)
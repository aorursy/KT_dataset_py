import numpy as np

#np is used as a reference for the numpy package. We can use any function of numpy through np.
fahrenheit =  [32,56,12,90,33,53]



#Lets simply print the fahrenheit to view the datatype of the elements in the list

print(fahrenheit)

print(type(fahrenheit))



#Lets view the same list in the numpy array

np_fahrenheit = np.array(fahrenheit)

print(np_fahrenheit)

print(type(np_fahrenheit))
#We will perform some operation on the List and Numpy arrays to see the difference in operations

celsius = (fahrenheit - 32)*(5/9)

print(celsius)
#Performing the same operation in Numpy

np_celsius = (np_fahrenheit - 32)*(5/9)

print(np_celsius)
#We will add two lists in this example.

fahrenheit_add = fahrenheit+fahrenheit

print(fahrenheit_add)
#Let's perform the same operation over the Numpy array

np_fahrenheit_add = np_fahrenheit + np_fahrenheit

print(np_fahrenheit_add)
print(np_fahrenheit[3])
print(np_fahrenheit[1:5])
print(np_fahrenheit > 34)
print(np_fahrenheit[np_fahrenheit > 34])
np.mean(np_fahrenheit[:])
np.median(np_fahrenheit[:])
np.std(np_fahrenheit[:])
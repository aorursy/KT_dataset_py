#Question 1

#Write a Python program using Scikit-learn to print the keys, number of rows-columns, feature names and the description of the Iris data.



import pandas as pd

data=pd.read_csv("../input/iris/Iris.csv")

print("\nKeys of Iris dataset:")

print(data.keys())

print("\nNumber of rows and columns of Iris dataset:")

print(data.shape) 
#Question 2

#Write a Python program to get the number of observations, missing values and nan values.



data.info()
#Question 3

#Write a Python program to create a 2-D array with ones on the diagonal and zeros elsewhere.



import numpy as np

matrix=np.eye(5)

print("NumPy array:\n", matrix)
#Question 4

#Write a Python program to load the iris data from a given csv file into a dataframe and print the shape of the data, type of the data and first 3 rows.



print("Shape of the data:")

print(data.shape)

print("\nData Type:")

print(type(data))

print("\nFirst 3 rows:")

print(data.head(3))
# Comment
print("Hello World")
print("Welcome to Kaggle")
# Variables
A = 10
B = "kaggle"
print (A, B)
# In python we don't need to declare variable type
# Python understand autometically
a = "Chaitanya"
b = "Python"
print(a + b) # Concadination of String
# Python is indentation for block of code
# Input from user
n1 = input("Enter a number ")
print(n1)
# By default input function give you a string type
type(n1)
#In Python2 input() used for integer number input
#In Python2 raw_input() used for string input
A = 10
B = 10.65
C = 10 + 6j
print(A)
print(B)
print(C)
# Python cannot differentiate between single and double quotes
A = 'Welcome to Kaggle'
B = "Python is Great"
print(A)
print(B)
# Tuples consists of a number of values separated by comma.
# It is enclosed within parenthesis
# A Tuple can have objects of different data types
A = (1, 2, 3.15, 'kaggle')
print(A)
A = [1, 2, 3.15, 'Kaggle']
print(A)
# Age and Name are Keys
# 24 and Chaitanya are the values associated with keys
A = {'Age':24, 'Name':'Chaitanya'}
print(A)
# 3 only appeared once
# You can also create a set by calling an in build-function 'set'
A = {1, 2, 3, 3}
B = [4, 5, 6, 6, 6, 4, 5, 7]
print(A)
print(set(B))
# Arithmetic Operators
A = 12
B = 13
print(A + B) # Addition
print(A - B) # Subtraction
print(A * B) # Multiplication
print(A / B) # Division
print(A % B) # Modulus
print(A ** B) # Exponent
print(A // B) # Floor Division
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
# Python program for simple calculator 



# Function to add two numbers 

def add(num1, num2): 

	return num1 + num2 



# Function to subtract two numbers 

def subtract(num1, num2): 

	return num1 - num2 



# Function to multiply two numbers 

def multiply(num1, num2): 

	return num1 * num2 



# Function to divide two numbers 

def divide(num1, num2): 

	return num1 / num2 



print("Please select operation -\n" , "1. Add\n", "2. Subtract\n", "3. Multiply\n", "4. Divide\n") 





# Take input from the user 

select = int(input("Select operations form 1, 2, 3, 4 :")) 



number_1 = int(input("Enter first number: ")) 

number_2 = int(input("Enter second number: ")) 



if select == 1: 

	print(number_1, "+", number_2, "=", 

					add(number_1, number_2)) 



elif select == 2: 

	print(number_1, "-", number_2, "=", 

					subtract(number_1, number_2)) 



elif select == 3: 

	print(number_1, "*", number_2, "=", 

					multiply(number_1, number_2)) 



elif select == 4: 

	print(number_1, "/", number_2, "=", 

					divide(number_1, number_2)) 

else: 

	print("Invalid input") 

def TowerOfHanoi(n , source, destination, auxiliary): 

    if n==1: 

        print ("Move disk 1 from source",source,"to destination",destination )

        return

    TowerOfHanoi(n-1, source, auxiliary, destination) 

    print ("Move disk",n,"from source",source,"to destination",destination )

    TowerOfHanoi(n-1, auxiliary, destination, source) 

          

n = 4

TowerOfHanoi(n,'A','B','C') 
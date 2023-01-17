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
#Basic Calculator

#Arithmatic functions

#ADDITION FUNCTION
def add(num1, num2):
    return (num1 + num2)

#SUBTRACTION FUNCTION
def subtract(num1, num2):
    return (num1 - num2)

#MULTIPLICATION FUNCTION
def multiply(num1, num2):
    return(num1*num2)

#DIVISION FUNCTION
def divide(num1, num2):
    
    if num2 == 0:     #HANDLING ZERO ERROR
        print("ERROR!!!! a number cannot be divided by 0\n")
    else:
        return (num1/num2)
    

#REMAINDER FUNCTION
def remainder(num1, num2):
    return (num1 % num2)


#CLEARS THE OUTPUT FOR NEXT SESSION
def clear():
    print("\n"*100)



    
def calculator():
    
    while True:
        num1 = int(input("Enter First Number\n"))
        num2 = int(input("Enter Second Number\n"))
        
        print("USE THE CODE SCHEMA")
        print("1.Addition")
        print("2.Subtraction")
        print("3.Multiplication")
        print("4.Division")
        print("5.Remainder")
        print("6.Close calculator")
        
        operation = int(input("ENTER THE OPERATION\n"))
    
    
    
        if operation == 1:
            print("\n\n\nAnswer Is: ", add(num1, num2))
            print("PRESS ENTER TO CONTINUE")
            x = input()
            clear()
        
        elif operation == 2:
            print("\n\n\nAnswer Is: ",subtract(num1, num2))
            print("PRESS ENTER TO CONTINUE")
            x = input()
            clear()
        
        elif operation == 3:
            print("\n\n\nAnswer Is: ",multiply(num1, num2))
            print("PRESS ENTER TO CONTINUE")
            x = input()
            clear()
        
        elif operation == 4:
            print("\n\n\nAnswer Is: ",divide(num1, num2))
            print("PRESS ENTER TO CONTINUE")
            x = input()
            clear()
        
        elif operation == 5:
            print("\n\n\nAnswer Is: ",remainder(num1, num2))
            print("PRESS ENTER TO CONTINUE")
            x = input()
            clear()
        
        elif operation == 6:
            print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nCALCULATOR SHUT DOWN")
            break
        
        else:
            print("INVALID CHOICE!!!!")
            print("PRESS ENTER TO CONTINUE")
            x = input()
            clear()
    
    


calculator()

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
# This function adds two numbers

def ad(x,y):

    return x+y



#This function subtracts two numbers

def sb(x,y):

    return x-y



#This function multiplies two numbers

def mt(x,y):

    return x*y



#This function divides two numbers

def dv(x,y):

    return x*y
#Take input from the user

print("Please, enter 2 numbers: ")



num1= int(input("Number 1: "))

num2= int(input("Number 2: "))

print("Please select one of the options")



option = int(input("[1] Addition \n [2] Subtraction \n [3] Multiplication  \n [4] Division \n"))



#Check if option is one of the four options

if option==1:

             print("Result [1] is: " + str(ad(num1, num2)))

                   

if option==2: 

             print("Result [2] is: " + str(sb(num1, num2)))

             

if option==3:

             print("Result [3] is: " + str(mt(num1, num2)))

             

if option==4:

             print("Result [4] is: " + str(dv(num1, num2)))

             
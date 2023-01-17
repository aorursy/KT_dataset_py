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
def ticket_type() :

    print("please enter 1 for First claas or 2 for second class ")

try :

    x=int(input("your Entry type is : "))

    if x==1  :

        print ("your ticket is :  FIRST CLASS\n\n" )

    elif x==2 :

        print ("your ticket is :  ECONOMY\n\n" )

    elif x>2 or x<1:

        print("wrong entry , please inter just numbers between 1 or 2 ]\n")

except ValueError :

    print("wrong entry , please inter just numbers between 1 or 2 \n")

def flight_type() :

    print("please enter R/r for Riyadh or J/j for Jeddah flight ")

z =input("your Entry is : ")

if z in ["r","R"] :

    print("your flight is : Riyadh\n")

elif z in ["J","j"] :    

        print("your flight is : Jeddah\n")

else :

    print("wrong entry , please inter just character R/r for Riyadh or J/j for Jeddah flight \n")
def num_of_people() :

    print("Enter number of Adults :")

try:

    adult_num=int(input(" your Entry  is :  "))

    if adult_num >=0 :

        print ("") 

    elif adult_num <0 :

        print ("\n please enter value >= 0 for adult number") 

except :

    print ("\n please enter *numerical* value >= 0 for adult number")   

    print("Enter number of Childs :   ")

try:

    child_num=int(input(" your Entry  is :  "))

    if child_num >=0 :

        print ("")

    elif child_num < 0 :

        print("\n please enter value >= 0 for child number ") 

except :

    print ("\n please enter *numerical* value >= 0 for child number")  

    print("Enter number of Infant :  " )

try:

    infant_num=int(input(" your Entry  is :  "))

    if infant_num >=0 :

        print ("") 

    elif infant_num <0 :

        print ("\n please enter value >= 0 for infant number") 

except :

    print ("\n please enter *numerical* value >= 0 for infant number")  
def personal_info () :

    print ("Please enter your First Name")   

first_name=input("First name is : ")

if first_name.isalpha():

    print (first_name)  

else :

        print ("please enter your name As a text value !\n")



print ("Please enter your Family Name") 

family_name=input("Family name is : ")

if family_name.isalpha():

    print(family_name)

else :

        print ("please enter your family name As a text value !\n")

# ID (just numbers):

print ("please enter your ID number")

try:

    cl_id=int(input("your ID is : "))

except :

    print("please enter numerical value of ID")
#contact number (just 10 digits):

print("please enter contact number  ")

contact_number=str(input("your entry is :"))

x=len(contact_number)

if x==10 :

    print ("your contact number is : ", contact_number )

else :

    print ("please enter numrical value of your contact number that consist of 10 digits !!")

#implementing functions  :

flight_type()

ticket_type()

num_of_people()

personal_info ()
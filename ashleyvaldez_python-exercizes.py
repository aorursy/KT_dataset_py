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
#Exercize 1

textValue = input("Enter your first name: ")

greeting = 'Hello {}.'.format(textValue)

print(greeting)
#Exercize 2

firstName = input("Enter your first name: ")

surName = input("Enter your surname: ")

greeting = 'Hello {}.'.format(firstName + surName)

print(greeting)
#Exercize 3

print("What do you call a bear with no teeth? A gummy bear!")
#Exercize 4

numValue = int(input("Enter one whole number: "))

numValue2 = int(input("Enter another whole number: "))

answer = numValue + numValue2

print(answer)                    
#Exercize 5

num1 = int(input("Enter your first whole number: "))

num2 = int(input("Enter your second whole number: "))

num3 = int(input("Enter your third whole number: "))

answer = (num1 + num2) * num3

print("it is...")

print(answer)
#Exercize 6

start = int(input("How many pizza slices did you start with?: "))

slicesEaten = int(input("How many pizza slices have you eaten?: "))

answer = start - slicesEaten

print(answer)
#Exercize 7

name = input("What's your name?: ")

age = int(input("Enter your current age: "))

answer = age + 1

print (name,"next birthday you will be", answer)
#Exercize 8

start = int(input("Enter bill amount: "))

dinersPresent = int(input("How many diners?: "))

answer = start/dinersPresent

print(answer)
#Exercize 9

days = int(input("How many days?: "))

hours = days * 24

minutes = hours * 60

seconds = minutes * 60

print(hours)

print(minutes)

print(seconds)
#Exercize 10

start = int(input("Enter a weight in kilograms: "))

pounds = start*2.20462

print(pounds)
#Exercize 11

start = int(input("Enter a number over 100: "))

then = int(input("Enter a number under 10: "))

answer = start/then

print("The answer is", answer)
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
#Exercize 12

num1 = int(input("Enter your first number: "))

num2 = int(input("Enter your second number: "))

if num1 > num2:

    print(num2, num1)

else:

    print(num1, num2)
#Exercize 13

num = int(input("Enter a number under 20: "))

if num >= 20:

    print("Too high!")

else:

    print("Thank you! :)")
#Exercize 14

num = int(input("Enter a number between 10 and 20: "))

if num>=10 or num<=20:

    print("Thank you! :)")

else:

    print("Incorrect answer! :(")
#Exercize 15

start = input("Enter your favorite color: ")

if start == "red" or start == "RED" or start == "Red":

    print("I like red too! :)")

else:

    print("I don't like",start,'I prefer red.')
#Exercize 16

question = input("Is it raining outside?: ")

question = str.lower(question)

if question== "yes":

    windy = input("Is it windy?: ")

    windy = str.lower(windy)

    if windy == "yes":

        print("It is too windy for an umbrella.")

    else:

        print("Take an umbrella")

if question != "yes":

    print("Enjoy your day!")
#Exercize 17

age = int(input("How old are you?: "))

if age>=18:

    print("You can vote.")

elif age==17:

    print("You can learn to drive.")

elif age==16:

    print("You can buy a lottery ticket.")

else:

    print("You can go Trick-or-Treating!")
#Excersize 18

num = int(input("Enter a number: "))

if num <10:

    print("Too low.")

elif num >= 10 and num <= 20:

    print("Correct!")

else:

    print("Too big!")
#Exercize 19

num = int(input("Enter a 1, 2, or 3: "))

if num==1:

    print("Thank you!")

elif num==2:

    print("Well done!")

elif num==3:

    print("Correct!")
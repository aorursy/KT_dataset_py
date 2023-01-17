# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

brics = pd.read_csv("../input/basics/brics.csv")

cars = pd.read_csv("../input/basics/cars.csv")
def your_choice(answer):

    if answer > 5:

        print("You are overaged.")

    elif answer <= 5 and answer >1:

        print("Welcome to the Toddler’s Club!")

    else:

        print("You are too young for Toddler’s Club.")



print(your_choice(6))

print(your_choice(3))

print(your_choice(1))

print(your_choice(0))
pizza = ["New York Style Pizza", "Pan Pizza", "Thin n Crispy Pizza", "Stuffed Crust Pizza"]



for choice in pizza:

    if choice == "Pan Pizza":

        print("Please pay $16. Thank you!")

        print("Delicious, cheesy " + choice)

    else:

        print("Cheesy pan pizza is my all-time favorite!")



print("Finally, I’m full!") 
fam = [1.73, 1.68, 1.71, 1.89]

for ele in enumerate(fam) :

    print("Tuple " , ele)

for index, height in enumerate(fam) :

    print("index " + str(index) + ": " + str(height))

for c in "family" :

    print(c.capitalize()) 
import pandas as pd

brics = pd.read_csv("../input/basics/brics.csv",index_col = 0)

print(brics)



for lab, row in brics.iterrows() :

    print("Label : ",lab)

    print("Row : ",row)

print("Extracting label and capital column from the row ")

for lab, row in brics.iterrows() : 

    print(lab + ": " + row["capital"])

counter = 0



while (counter < 10):

    print('The count is:' , counter)

    counter = counter + 1



print("Done!")

f = lambda x, y : x + y

f(1,1)
Celsius = [39.2, 36.5, 37.3, 37.8]

Fahrenheit = map(lambda x: (float(9)/5)*x + 32, Celsius)

print(list(Fahrenheit))

fib = [0,1,1,2,3,5,8,13,21,34,55]

result = filter(lambda x: x % 2, fib)

print("Filtered List : ",list(result))

from functools import reduce

# Summing up a list using reduce with lambda function

print("Summing up a list using reduce with lambda function : ",reduce(lambda x,y: x+y, [47,11,42,13]))



# Determining the maximum of a list of numerical values by using reduce

f = lambda a,b: a if (a > b) else b

print("Determining the maximum of a list of numerical values by using reduce : ",reduce(f, [47,11,42,102,13]))
try:

   fh = open("testfile2", "w")

   fh.write("This is my test file for exception handling!!")

except IOError:

   print("Error: can\'t find file or read data")

else:

   print("Written content in the file successfully")

   fh.close()

try:

    fh = open("testfile", "w")   

    fh.write("This is my test file for exception handling!!")

    print("File Written Succesfully \n")

finally:

    fh.close()

    print ("Finally clause : Postprocessing completed \n")

try:

   fh = open("testfile3", "r")

   fh.write("This is my test file 3 for exception handling!!")

except:

    print("Error : can\'t find file or read data")

finally:

   print("Finally section ")

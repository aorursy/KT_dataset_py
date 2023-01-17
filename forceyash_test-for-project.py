# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd



df1 = pd.read_csv("../input/Live Test 1.csv")

df2 = pd.read_csv("../input/Live Test 2.csv")



#1.	Get the number of rows and columns in the imported file



print(df1.shape)

print(df2.shape)



# 2.Get the top 15 rows of the imported data



print(df1.head(15))

print(df2.head(15))
#3.	Get the bottom 10 rows of the imported data



print(df1.tail(15))

print(df2.tail(15))
# 4.	Get the list of variables, variable data type, and variable not-null count

print(df1.info())

print(df2.info())
#5.	Get univariate distribution of numeric variables

df1["Age"].value_counts().plot.bar()



df1["Height"].value_counts().plot.bar()
df2["Weight"].value_counts().plot.bar()
#7.	Remove duplicate “id” from the dataset and assign the resulting dataset as a new dataset

df1.drop_duplicates(subset=['id'], keep=False)

df2.drop_duplicates(subset=['id'], keep=False)

#8.	Rename column “name” to “Person_Name”

df1.rename(columns = {"Name": "Person_Name"})



#4.	Left join left test 1 with live test 2 and get variable “weight” in live test 1

df = pd.merge(df1, df2, left_on = "id", right_on = "id")

df
#5.	Export and save the file as “live Test 3.csv”

df.to_csv("live Test 3.csv")
#6.	Drop column “id” from the dataset

df1.drop(columns = "id")

df2.drop(columns = "id")
#7.	Get average weight by Gender

def avg(df, gen):

    x = df[df.Gender == gen]

    x = x.dropna()

    print(x["Weight"].sum()/x["Weight"].count())

    

avg(df, "M")

avg(df, "F")

    

def avg_h(df, gen):

    x = df[df.Gender == gen]

    print(x["Height"].sum()/x["Height"].count())



num = 7

factorial = 1

if num < 0:

   print("Sorry")

elif num == 0:

   print("The factorial of 0 is 1")

else:

   for i in range(1,num + 1):

       factorial = factorial*i

   print("The factorial of",num,"is",factorial)



#Using while loop

num = 7

x = num

factorial = 1

if num < 0:

   print("Sorry")

elif num == 0:

   print("The factorial of 0 is 1")

else:

   while(num>1):

    factorial = factorial*num

    num =num-1    

   print("The factorial of",x,"is",factorial)
def recur(n):

   if n == 1:

       return n

   else:

       return n*recur(n-1)

    

recur(7)
[recur(x) for x in range(0,8) if x>1]
l = [91, 46, 2, 102, 44]



l.sort()

l[-2]
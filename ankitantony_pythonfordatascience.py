#Addition

print(3 + 5)



#Subtraction

print(8 - 3)



#Multiplication

print(2 * 4)



#Division

print( 35 / 7)
# cannot add string to int, error will be thrown

# "Ankit" + 25



# Output:

#---------------------------------------------------------------------------

#TypeError                                 Traceback (most recent call last)

#<ipython-input-50-a824a373d323> in <module>

#      1 # cannot add string to int, error will be thrown

#----> 2 "Ankit" + 25

#

#TypeError: must be str, not int

"Ankit" + "20"
"Ankit" * 5
# greater than

print(45 > 43)



# less than

print(56 < 23)



print(34 * 34 < 45 * 25)



print(36 == 36)
# and operator

print(0 and 3)



print(3 and 0)



print(3 and 5)



# or operator

print(0 or 3)



print(3 or 0)



print(3 or 5)
a = 5

A = 3

print(a)

print(A)
a = 7

b = a

a = 3

print(a)

print(b)
a = 5

print(a)

_a = 3

print(_a)

#invalid naming

# @a = 9

# 1a = 2 
a = 5

type(a)
b = "Ankit"

type(b)
x = 8

if(x % 2 == 0):

    print("Even")

else:

    print("Odd")

    

y = 3

if(y % 2 == 0):

    print("Even")

else:

    print("Odd")        
y = 62



if(y > 90):

    print("Grade A")

elif(y > 60):

    print("Grade B")

else:

    print("Grade F")
for i in range(11, 50):

    print(i)
for i in range(11, 50, 2):

    print(i)
def compare(a, b):

    if(a > b):

        greater = a

    else:

        greater = b

    

    return greater          



compare(25, 15)
# create a list

marks = [56, 63, 45, 68, 91]

marks
# index of list starts with 0

marks[3]
marks[0:]
marks[0:3] # element at index 3 is excluded
# add an element to marks list

marks.append(58)

marks
# add multiple elements to the marks list

marks.extend([49, 85])

marks
# add a list of elements to the existing list

marks.append([45, 36])

marks
# removing elements

marks.remove([45, 36])

marks
del marks[3]

marks
# accessing the list elements

for mark in marks:

    print(mark)
# increment all marks by 1

for mark in marks:

    print(mark+1)
# create a dictionary (key: value pair)

marks = {'History':45, 'Geography':54, 'Hindi':56}

marks
# accessing elements in a dictionary

marks['History']
# adding new elements to the dictionary

marks['English'] = 85

marks
# updating elements in the dictionary

marks.update({'History':49})

marks



marks['English'] = 87

marks
# updating the dictionary with multiple elements

marks.update({'English':91, 'Hindi': 73})

marks
del marks['English']

marks
# reading CSV file using pandas

import pandas as pd

data_CSV = pd.read_csv("../input/pythonfordatasciencedataset/data-CSV.csv")

data_CSV.head()
# reading excel file using pandas

df_excel = pd.read_excel('../input/pythonfordatasciencedataset/data-Excel.xlsx')

df_excel.head()
# reading the data for dataframe

data_DFBasics = pd.read_csv("../input/pythonfordatasciencedataset/data-DFBasics.csv")

data_DFBasics
# reading the dimensions of the dataframe data_DFBasics

data_DFBasics.shape
# print top 5 rows of DF data_DFBasics

data_DFBasics.head()
# print bottom 5 rows of DF data_DFBasics

data_DFBasics.tail()
# print bottom 10 rows

data_DFBasics.tail(10)
# reading column names

data_DFBasics.columns
# selecting a single column

data_DFBasics['Name']
# selecting multple columns using names

data_DFBasics[['PassengerId', 'Survived']]
# importing data for indexing example

data_Indexing = pd.read_csv("../input/pythonfordatasciencedataset/data-Indexing.csv")

data_Indexing.head()
data_Indexing['Bare_Nuclei']
# selecting rows by their positions

data_Indexing.iloc[:5]
# selecting rows by their positions

data_Indexing.iloc[:,:5]
data_Indexing[data_Indexing['Cancer_Type'] == 2]
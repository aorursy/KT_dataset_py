import pandas as pd # Datensets

import numpy as np # Data Manipulation
# Read in all data

data = pd.read_csv("../input/avocado.csv")
# Create Sample Dataset



testcolumns = ["Date", "Price", "Name", "Volume", "42"]



testdata = [

    ["01-01-2019", 100, "Redhat", 20, 42],

    ["02-01-2019", 200, "Pop", 10, 42],

    ["03-01-2019", 300, "Mint", 5, 42],

    ["04-01-2019", 400, "Arch", 2.5, 42],

    ["05-01-2019", 500, "Suse", 1.25, ]

]



testdata = pd.DataFrame(columns = testcolumns, data = testdata)
# ToDo: Read out schema of the Dataset
# This is how the testdata looks:

print(testdata.columns.values)

testdata.head()
# ToDo: print column names of data



# ToDo: print last column name



# ToDo: print first 10 rows of data
# Example

testdata = testdata.drop(["Name"], axis = 1)

print("missing data\n", testdata.isnull().sum())

testdata = testdata.dropna(how = "any")

testdata.head()
# ToDo: delete "Unnamed: 0"



# ToDo: Read out missing data



# ToDo: delete rows with missing Data
print("before transformation\n", testdata["Date"].head())

testdata["Date"] = pd.to_datetime(testdata["Date"])

print("after transformation\n", testdata["Date"].head())
# Overview of Date Data

# Convert Date to Datetime

# New Overview of Data

# example

testdata["meaningless"] = testdata["Price"] + testdata["Volume"]

testdata["just dividing by 5"] = testdata["Volume"] / 5

testdata.head()
# ToDo: Add "Total Sales" to Dataset(Price * Volume)



# ToDo: Add "real Price" to Dataset (Price corrected by inflation)



# 42 zu ersetzen:

a1 = [1, 42] #Ergebnis 1

a2 = [2, 42] #Ergebnis 2

a3 = [3, "42"] #Ergebnis 3

a4 = [4, 42] #Ergebnis ...

a5 = [5, "42"]

a6 = [6, 42]

a7 = [7, "42"]

a8 = [8, 42]

antworten = [a1,a2,a3,a4,a5,a6, a7, a8]

meine_antworten = pd.DataFrame(antworten, columns = ["Id", "Category"])

meine_antworten.to_csv("meine_loesung_Aufgaben1.csv", index = False)
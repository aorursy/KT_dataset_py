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

data.shape
# This is how the testdata looks:

print(testdata.columns.values)

testdata.head()
# ToDo: print column names of data

print(data.columns.values)



# ToDo: print last column name

print(data.columns.values[-1])

# ToDo: print first 10 rows of data

print(data.head(10))

print(sum(data.head(10).AveragePrice))
# Example

testdata = testdata.drop(["Name"], axis = 1)

print("missing data\n", testdata.isnull().sum())

testdata = testdata.dropna(how = "any")

testdata.head()
# ToDo: delete "Unnamed: 0"

data.drop(["Unnamed: 0"], axis = 1, inplace=True)

# ToDo: Read out missing data

print(data.isnull().sum() )

# ToDo: delete rows with missing Data

data.dropna(how="any", inplace=True)

data.shape
print("before transformation\n", testdata["Date"].head())

testdata["Date"] = pd.to_datetime(testdata["Date"])

print("after transformation\n", testdata["Date"].head())
# Overview of Date Data

data.Date.describe()
# Convert Date to Datetime

data.Date = pd.to_datetime(data.Date)
# New Overview of Data

data.Date.describe()
# example

testdata["meaningless"] = testdata["Price"] + testdata["Volume"]

testdata["just dividing by 5"] = testdata["Volume"] / 5

testdata.head()
# ToDo: Add "Total Sales" to Dataset(Price * Volume)

data["Total Sales"] = data["AveragePrice"] + data["Total Volume"]

# ToDo: Add "real Price" to Dataset (Price corrected by inflation)

data["real Price"] = data["AveragePrice"] * 1 + (0.02015 * (2018 - data["year"]))

data.head()
# 42 zu ersetzen:

a1 = [1, 18249] #Ergebnis 1

a2 = [2, 14] #Ergebnis 2

a3 = [3, "region"] #Ergebnis 3

a4 = [4, "2015-12-20"] #Ergebnis ...

a5 = [5, 11.29]

a6 = [6, 18243]

a7 = [7, "2018-03-25 00:00:00"]

a8 = [8, 1.340]

antworten = [a1,a2,a3,a4,a5,a6, a7, a8]

meine_antworten = pd.DataFrame(antworten, columns = ["Id", "Category"])

meine_antworten.to_csv("meine_loesung_Aufgaben1.csv", index = False)
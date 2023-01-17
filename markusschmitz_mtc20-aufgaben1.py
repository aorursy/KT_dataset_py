import pandas as pd # Datensets

import numpy as np # Data Manipulation
# Read in all data

data = pd.read_csv("../input/testsets/avocado.csv")
# ToDo: read the data of september and october



# ToDo: append the data of october to september to combine the data frames

# ToDo: Join the "fahrten" and "wetter" data from december to combine the data frames

# ToDo: Read out schema of the Dataset
# ToDo: print column names of data



# ToDo: print last column name



# ToDo: print first 10 rows of data
# ToDo: Select only the Average price from the dataset



# ToDo: Select S, L and XL from the dataset

# ToDo: Select all rows where region is "Chicago"



# ToDo: Select all rows where the AveragePrice is over 1.00 and the region is "Boston"

#ToDo: Describe the Data

# ToDo: get the maximum Price of the last 10 entries



# ToDo: get the std of the AveragPrice for all Entries with more than 30000 sold bags

# ToDo: delete "Unnamed: 0"



# ToDo: Read out missing data



# ToDo: delete rows with missing Data



# ToDo: Count new rows
# ToDo: Print overview of Date Data with describe()



# ToDo: Convert Date to Datetime



# ToDo: Print overview of Date Data again

# ToDo: Add "TotalSales" to Dataset(Price * Volume)



# ToDo: Add "RealPrice" to Dataset (Price corrected by inflation)

# Nummer 0 durch Lösungen ersetzen:



a1 = [1, 42]             

a2 = [2, 42]                

a3 = [3, "42"]     

a4 = [4, 42]      

a5 = [5, 42]                 

a6 = [6, 42]                 

a7 = [7, 42]      

a8 = [8, 42]   

a9 = [9, 42]

a10 = [10, 42]

a11 = [11, 42]



antworten = [a1,a2,a3,a4,a5,a6, a7, a8, a9, a10, a11]

meine_antworten = pd.DataFrame(antworten, columns = ["Id", "Category"])

meine_antworten.to_csv("meine_loesung_Aufgaben1.csv", index = False)
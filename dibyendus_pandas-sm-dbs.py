import pandas as pd # import pandas Library

import numpy as np
print(pd.Series([1,2,3]))

ser = pd.Series([1,2,3] , index = ['first', 'second','third']) #Index acts as reference 

print(se)
ser['first'] #Using index to access Data elements
se[0] #Using postion to access data elements
# Python Dictionary

p_dict = {'name' : ['Sachin','Sourav','Rahul'], 

         'country' : ['Ind','Ind','Ind'],

         'Runs': ['10000','10500','10100']}

print(p_dict)
#Dataframe - This will convert the dictionary to DataFrame

df = pd.DataFrame(p_dict) 

print(df.head())  #by default shows the first five rows

print('\n')

print(df.head(2))
df.tail(2) #last two rows
#Selecting a specific column

#Below are the two ways which produces the same result



print(df.Runs)

print(df['Runs'])
print(df.iloc[0]) #First row of the dataframe

print('\n')

print(df.iloc[-1]) #Only the last row

print('\n')

print(df.iloc[ :,1]) #all the rows and first column
print(df.iloc[ :,0:2]) #All rows and 1st to 2nd column , : -> this will retrieve all rows/columns
df = pd.read_csv('train.csv') # read_csv is the function to read csv file
df.shape # To check the count of rows and columns
df.columns #to check the column names
df.dtypes # to see datatypes of the columns
df["Survived"].value_counts() # to check unique values present in a column
# No. of females who survived?

df[(df.Sex =='female') & (df.Survived ==1)].shape[0] #() are mandatory for multiple conditions,()head -first 5 rows
# % of Male survivors

(len(df[(df.Sex =='male') & (df.Survived ==1)])/len(df))*100
# % of Female survivors

(len(df[(df.Sex =='female') & (df.Survived ==1)])/len(df))*100
# To check if there is any duplicate PassengerID

sum(df.PassengerId.duplicated())
#Average Age of passengers grouped by Gender and Survival Status

df.groupby(by = ['Sex','Survived']).mean().Age

import numpy as np 
import pandas as pd
import os

#Import csv file
df=pd.read_csv("../input/spacex_launch_data.csv")
df.head()

#Lets find the dimension of the dataframe
df.shape
# There is 57 rows and 11 cloumns
#Lets print the column name
df.columns
# Lets find the no of unique element in each column

#First lets store the column name into a list variable
column=list(df.columns)

#Lets print unqiue elemnt in each column
for i in column:
    print("Column name: "+str(i))
    print("Total unique Elements: "+str(len(df[i].unique())))
    print("Unique List are: ")
    print(df[i].unique())
    print("*************************************************************************************************")

# You can feel why I am doing like this.Insted of this,
# we can put in  beautiful data visualization.For a chance I want to see like this.
# Lets comapre with year.Now lets extract year from the date.

year=[]
for i in df["Date"]:
    year.append(i.split("-")[0])
print("year")
print(year)
print("Unique Year")
print(set(year))
#Customer diversity

#Top repeated customer for spaceX and these are the only customer who came more than one time 
df["Customer"].value_counts()[:9]
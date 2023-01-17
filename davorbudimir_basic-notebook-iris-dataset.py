# import exploration files 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

# read in data
file_path = '../input/irisdataset/Iris.csv'
data = pd.read_csv(file_path)

############################################################################## 
#Data Exploration
##############################################################################

#rows and columns returns (rows, columns)
print("Rows and columns returns: \n",data.shape)

#returns the first x number of rows when head(num). Without a number it returns 5
print("Data head: \n ", data.head())

#returns the last x number of rows when tail(num). Without a number it returns 5
print("Data tail: \n", data.tail())

#returns an object with all of the column headers 
print("Column headers: \n", data.columns)

#basic information on all columns 
print("Basic information: \n",data.info())

#gives basic statistics on numeric columns
print("Data describe: \n",data.describe())

#shows what type the data was read in as (float, int, string, bool, etc.)
print("Shows what type: \n",data.dtypes)

#shows which values are null
print("Values are null: \n",data.isnull())

#shows which columns have null values
print("Columns have null values: \n",data.isnull().any())

#shows for each column the percentage of null values 
print("Percentage of null values: \n",data.isnull().sum() / data.shape[0])

#plot histograms for all numeric columns 
data.hist()


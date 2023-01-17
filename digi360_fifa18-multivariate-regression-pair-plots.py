# (c) Sydney Sedibe, 2018

import warnings 
warnings.filterwarnings('ignore')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns # Seaborn for pairplots

# Set text size
plt.rcParams['font.size'] = 18

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

file_list = os.listdir("../input")
print(file_list)
odf = pd.read_csv("../input/" + file_list[1], low_memory=False) # odf = original dataframe with complete dataset
odf.columns
def showDetails(df):
    print("-------------------------------------------------------------------------------------------------------------------")
    print('{:>35}'.format("Shape of dataframe:") + '{:>12}'.format(str(df.shape)))
    containsNulls = "Yes" if df.isnull().any().any() else "No"
    print("Does dataframe contain null values: " + containsNulls)
    null_columns = df.columns[df.isnull().any()]
    print("Number of columns with null values: " + str(df[null_columns].isnull().any().sum()))
    null_rows = df[df.isnull().any(axis=1)][null_columns]
    print("Number of records with null values: " + str(len(null_rows)))
    print('{:>35}'.format("Percentage of null records:") + '{:>6.2f}'.format(len(null_rows) / len(df) * 100) + "%")
    print("-------------------------------------------------------------------------------------------------------------------")

showDetails(odf)
nv_df = odf[odf.isnull().any(axis=1)] # nv_df ==> null value dataframe
showDetails(nv_df)
nv_df.head()
wdf = odf[['Overall', 'Value', 'Wage', 'Aggression', 'Free kick accuracy', 'Sprint speed', 'Finishing']]
showDetails(wdf)
wdf.head()
def toFloat(string):
    """Function to convert Wage and Value strings to floats"""
    string = string.strip(" ")
    if string[-1] == 'M':
        return float(string[1:-1]) * 1000000
    elif string[-1] == 'K':
        return float(string[1:-1]) * 1000
    else:
        return float(string[1:])
wdf['Value'] = [toFloat(value) for value in wdf['Value']]
wdf['Wage'] = [toFloat(wage) for wage in wdf['Wage']]
wdf.head()
print("There are " + str(len(wdf[wdf["Wage"] == 0])) + " rows with a wage value of 0 in the Wage column")
print("There are " + str(len(wdf[wdf["Value"] == 0])) + " rows with a player-value of 0 in the Value column")
def replaceZeroValues(df, column):
    subset = df[ df[column] != 0 ][column]
    nonzero_mean = subset.mean()
    print("The nonzero_mean for " + column + " is " + str(nonzero_mean))
    df.loc[ df[column] == 0, column ] = nonzero_mean
    
replaceZeroValues(wdf, "Wage")
replaceZeroValues(wdf, "Value")
print("There are " + str(len(wdf[wdf["Wage"] == 0])) + " rows with a wage value of 0 in the Wage column")
print("There mininum value for the Wage column is " + str(wdf["Wage"].min()))
print("There are " + str(len(wdf[wdf["Value"] == 0])) + " rows with a player-value of 0 in the Value column")
print("There mininum value for the Value column is " + str(wdf["Value"].min()))
wdf.info()
def removeExtraChars(string):
    sc = "" #special character: either '+' or '-'
    if "+" in string:
        sc = "+"
    elif "-" in string:
        sc = "-"
    else:
        return int(string)
    return int(string[:string.find(sc)])

def cleanUpColumn(df, column):
    return [removeExtraChars(row) for row in df[column]]
wdf["Aggression"] = cleanUpColumn(wdf, "Aggression")
wdf["Free kick accuracy"] = cleanUpColumn(wdf, "Free kick accuracy")
wdf["Sprint speed"] = cleanUpColumn(wdf, "Sprint speed")
wdf["Finishing"] = cleanUpColumn(wdf, "Finishing")

wdf.info()
sns.pairplot(wdf);
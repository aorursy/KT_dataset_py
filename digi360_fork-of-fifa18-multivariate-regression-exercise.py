# (c) Sydney Sedibe, 2018

import warnings 
warnings.filterwarnings('ignore')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

file_list = os.listdir("../input")
print(file_list)
odf = pd.read_csv("../input/" + file_list[1], low_memory=False) # odf = original dataframe with complete dataset
odf.head()
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
def toFloat(string):
    """Function to convert Wage and Value strings to floats"""
    string = string.strip(" ")
    if string[-1] == 'M':
        return float(string[1:-1]) * 1000000
    elif string[-1] == 'K':
        return float(string[1:-1]) * 1000
    else:
        return float(string[1:])
odf["NumericalWage"] = [toFloat(x) for x in odf["Wage"]]
odf["NumericalValue"] = [toFloat(x) for x in odf["Value"]]
print("There are " + str(len(odf[odf["NumericalWage"] == 0])) + " rows with a wage value of 0 in the NumericalWage column")
print("There are " + str(len(odf[odf["NumericalValue"] == 0])) + " rows with a player-value of 0 in the NumericalValue column")
def replaceZeroValues(df, column):
    subset = df[ df[column] != 0 ][column]
    nonzero_mean = subset.mean()
    print("The nonzero_mean for " + column + " is " + str(nonzero_mean))
    df.loc[ df[column] == 0, column ] = nonzero_mean
replaceZeroValues(odf, "NumericalWage")
replaceZeroValues(odf, "NumericalValue")
print("There are " + str(len(odf[odf["NumericalWage"] == 0])) + " rows with a wage value of 0 in the NumericalWage column")
print("There are " + str(len(odf[odf["NumericalValue"] == 0])) + " rows with a player-value of 0 in the NumericalValue column")
odf["NumericalValue"].iloc[164]
# First, let's define a function to quickly scatter-plot two columns and draw a trendline that we can reuse
def myplot(x, y):
    """Draws a scatter plot of columns x and y"""
    ax = plt.scatter(x, y)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    m, c = z
    formula = ("y = " + str(m) + " x " + str(c)) if (c < 0) else ("y = " + str(m) + " x + " + str(c))
    print(formula)
    plt.title(x.name + " vs " + y.name)
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    plt.plot(x, p(x), "red")
    plt.show()
myplot(odf["Overall"], odf["NumericalValue"])
odf[odf['NumericalValue'] == 0] = odf['NumericalValue'].mean()
odf["LogValue"] = np.log(odf['NumericalValue'].astype('float64'))
logValueMax = odf["LogValue"].min()
logValueMax
myplot(odf["Overall"], odf["LogValue"])
stringRows = [x for x in odf["Sprint speed"] if ("+" in x) or ("-" in x)]
print("There are " + str(len(stringRows)) + " rows with special characters in the 'Sprint speed' column.")
stringRows
def removeExtraChars(string):
    sc = "" #special character: either '+' or '-'
    if "+" in string:
        sc = "+"
    elif "-" in string:
        sc = "-"
    else:
        return string
    return string[:string.find(sc)]
odf["SprintSpeed"] = odf["Sprint speed"].apply(removeExtraChars)
total = 0
[total+1 for rowValue in odf["SprintSpeed"] if "+" in rowValue or "-" in rowValue]
print(str(total) + " rows in the new SprintSpeed column contain +/-")
odf['SprintSpeed'] = odf['SprintSpeed'].astype('float')
odf.info()
myplot(odf["SprintSpeed"], odf["LogValue"])
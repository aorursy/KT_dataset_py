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
wdf = odf[['Name', 'Age', 'Overall', 'Potential', 'Value', 'Wage', 'Preferred Positions']]
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
wdf = odf[['Name', 'Age', 'Overall', 'Potential', 'Value', 'Wage', 'Preferred Positions']]
wdf['Value'] = [toFloat(value) for value in wdf['Value']]
wdf['Wage'] = [toFloat(wage) for wage in wdf['Wage']]
wdf.head()
def myplot(x, y):
    ax = plt.scatter(x, y)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    #plt.title('The Title')
    plt.plot(x, p(x), "red")
    plt.show()
myplot(x=wdf['Overall'], y=wdf['Value'])
wdf['LoggedValue'] = np.log(wdf['Value'])
wdf['LoggedValue'].tail()
#myplot(x=wdf['Overall'], y=wdf['LoggedValue'])
f, ax = plt.subplots(figsize=(10, 5))
x = wdf["Overall"]
y = wdf["LoggedValue"]
ax = sns.regplot(x, y, scatter_kws={"s": 10})
ax.set_xlabel("Overall")
ax.set_ylabel("Market Value")
ax.set_title("Player Overall vs Market Value")

# Trendline code still doesn't work for some reason I haven't figured out yet
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x, p(x), "red")
plt.show()
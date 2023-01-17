# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import pandas as pd

print("All pkgs have been imported successfully.")
dfXRP = pd.read_csv(r'../input/xrpusd/XRP.csv')

print("Displaying dataset")

dfXRP
print("Display detailed info about the dataset")

dfXRP.info()
nRow,nCol = dfXRP.shape



nRow, nCol = dfXRP.shape

print("Number of columns in the dataset =",nCol)

print("Number of rows in the dataset    =",nRow)

print("Is there any null value in the dataset ?",dfXRP.isnull().values.any())

print("Display a column-wise null value count\n")

dfXRP.isnull().sum()
print("Getting rid of the null data")

dfXRP.dropna(inplace=True)

dfXRP.head()
print("Number of null values dropped =",nRow - len(dfXRP))
datesLen = nRow

#print(nRow)

datesArr = []

datesArr = [1 for i in range(datesLen)]





# Making the x-values for the dates

for i in range(0,837):

    datesArr[i] = i+1

    

##print(datesArr,"\n")





    

x = dfXRP['Open'].values

x1 = dfXRP['Close'].values

x2 = dfXRP['High'].values

x3 = dfXRP['Low'].values



print("These are the open values")

print(x,"\n")



print("These are the close values")

print(x1,"\n")



print("These are the high values")

print(x2,"\n")



print("These are the low values")

print(x3,"\n")



print("These are the volume XRP values")



y = dfXRP['Volume'].values

print(y,"\n")



print("These are the adjusted close values")



y1 = dfXRP['Adj Close'].values

print(y1,"\n")
stdDevArray = [0 for i in range(nRow)]

annualVolatility = [0 for i in range(nRow)]

numberOfDays = 252*(nRow/365) #this is scaled since 252 days is the number of tradeable days per year. Find the number of years this data took place and

#scale accordingly



for i in range(1,len(x1)):

    stdDevArray[i-1] = ((x1[i]/x1[i-1]) - 1)

    annualVolatility[i-1] = stdDevArray[i-1]*np.sqrt(numberOfDays)



# print("Here is the Std Deviation Array \n")

# print(stdDevArray,"\n")



#print("Here is the Annual Volatility Array \n")

#print(annualVolatility,"\n")



# daysTotal = len(annualVolatility)

# years = daysTotal/365



#print("This is plotted over the course of" + years + "years.")
plt.figure(figsize=(20,25))

plt.subplot(4,2,1)

plt.plot(datesArr,x, color='green')

plt.ylabel("Open currency values")

plt.xlabel("Number of Days")

plt.title("Prices of XRP over time")





plt.subplot(4,2,2)

plt.plot(datesArr,x1, color='red')

plt.xlabel("Number of Days")

plt.ylabel("Close Prices of XRP")

plt.title("Closing Prices of XRP charted over time")





## Tried overlaying the opening and closing price on the same graph, but the prices are so close that it looks like a shadow.

plt.subplot(4,2,3)

plt.plot(datesArr,x, color='green')

plt.plot(datesArr,x1, color='red')

plt.xlabel("Number of Days")

plt.ylabel("Open and Close Prices of XRP")

plt.title("Composite Graph of XRP prices charted over time")



plt.subplot(4,2,4)

plt.plot(datesArr,annualVolatility, color='orange')

plt.xlabel("Number of Days")

plt.ylabel("Volatility Value of XRP")

plt.title("Volatility over time plot")

plt.suptitle("Prices and Volatility of vs. Time")



plt.subplot(4,2,5)

plt.plot(datesArr,annualVolatility, color='blue')

plt.plot(datesArr,x1)

plt.xlabel("Number of days")

plt.ylabel("Volatility of XRP/Closing Prices of XRP")

plt.title("Volatility of XRP & Closing XRP values vs Time")



# plt.subplot(4,2,6)

# plt.plot(x2,y1, color='indigo')

# plt.xlabel("High currency values")

# plt.ylabel("Volume of Currency")

# plt.title("Volume of Currency vs High currency")

# plt.suptitle("Volume of Ripple and Currency vs Open Currency")



# plt.subplot(4,2,7)

# plt.plot(x3,y, color='violet')

# plt.xlabel("Low currency values")

# plt.ylabel("Volume of XRP")

# plt.title("Volume of XRP vs Low currency")

# plt.suptitle("Volume of Ripple and Currency vs Open Currency")



# plt.subplot(4,2,8)

# plt.plot(x3,y1, color='brown')

# plt.xlabel("Low currency values")

# plt.ylabel("Volume of Currency")

# plt.title("Volume of Currency vs Low currency")

# plt.tight_layout()

# print("Volume of Ripple and Currency vs Currency types")
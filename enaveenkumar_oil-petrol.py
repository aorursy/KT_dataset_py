# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Importing required Libraries which includes reading, for doing EDA and visualization

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib

from matplotlib import pyplot as plt
#Reading the Data file 

Price_yearly = pd.read_csv("../input/Spot_Prices.csv")
#Checking the first 10 rows 



Price_yearly.head()
Price_yearly.columns
#The initial data analysis shows that Crude Oil is measured in Barrel so as price. 

#To make consistent for our analysis, COnverting the Barrel to gallon by dividing the factor 42

Price_yearly['Europe Brent Spot Price FOB /gal']=Price_yearly['Europe Brent Spot Price FOB ($/bbl) $/bbl']/42

Price_yearly['Cushing OK WTI Spot Price FOB /gal']=Price_yearly['Cushing OK WTI Spot Price FOB ($/bbl) $/bbl']/42
#Removing Europe Brent Spot Price FOB ( /bbl)/bbl) /bbl	 and Cushing OK WTI Spot Price FOB ( /bbl)/bbl) /bbl as we have replaced with gallon



Price_yearly=Price_yearly.drop('Europe Brent Spot Price FOB ($/bbl) $/bbl',axis=1)

Price_yearly=Price_yearly.drop('Cushing OK WTI Spot Price FOB ($/bbl) $/bbl',axis=1)
#Cloumn Month data type changing

Price_yearly['Month'] = pd.to_datetime(Price_yearly['Month'],format='%b-%y')
Price_yearly.head()
#Basic analysis, stats on the data

Price_yearly.describe()
Price_yearly.columns
# Checking the monthly trend - Univariate analysis



for col in Price_yearly.columns[1:len(Price_yearly.columns)]:

    sns.boxplot(x='Month', y=col, 

                palette=['b','r'], data=Price_yearly)

    sns.despine(offset=10, trim=True)



    plt.show()
lag=Price_yearly['U.S. Gulf Coast Kerosene-Type Jet Fuel Spot Price FOB ($/gal) $/gal'].shift(1)



Price_yearly['U.S. Gulf Coast Kerosene-Type Jet Fuel Spot Price FOB ($/gal) $/gal- Monthly_Change']=(Price_yearly['U.S. Gulf Coast Kerosene-Type Jet Fuel Spot Price FOB ($/gal) $/gal']-lag)*100/lag
Price_yearly['U.S. Gulf Coast Kerosene-Type Jet Fuel Spot Price FOB ($/gal) $/gal- Monthly_Change']
lag1=Price_yearly['Los Angeles CA Ultra-Low Sulfur CARB Diesel Spot Price ($/gal) $/gal'].shift(1)



Price_yearly['Los Angeles CA Ultra-Low Sulfur CARB Diesel Spot Price ($/gal) $/gal Month-Change']=(Price_yearly['Los Angeles CA Ultra-Low Sulfur CARB Diesel Spot Price ($/gal) $/gal']-lag1)*100/lag1
Price_yearly.assign(New=(Price_yearly < Price_yearly.shift()).all(1).astype(int))
Pric
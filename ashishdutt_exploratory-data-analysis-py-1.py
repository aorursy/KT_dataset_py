# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Loading the Death Records data, manner of death data & day of week death data
deathRec=pd.read_csv("../input/DeathRecords.csv")
deathBy=pd.read_csv("../input/MannerOfDeath.csv")
deathDay=pd.read_csv("../input/DayOfWeekOfDeath.csv")
#Determine the number of rows and columns in the dataset
print ("\nDeath Records (row x col) ",deathRec.shape)
print ("\nDeath caused (row x col) ",deathBy.shape)
print ("\nDay of death (row x col) ",deathDay.shape)
# Print the column headers/headings
names=deathRec.columns.values
print ("\n Death Records data headers\n")
print (names)
names=deathBy.columns.values
print ("\n Death caused by headers\n")
print (names)
names=deathDay.columns.values
print ("\n Day of death headers\n")
print (names)
# print the rows with missing data
print ("The count of rows with missing values in Death Records data: \n", deathRec.isnull().sum())
print ("\nThe count of rows with missing values in Death caused by data: \n", deathBy.isnull().sum())
print ("\nThe count of rows with missing values in Death Day data: \n", deathDay.isnull().sum())
# Show the Frequency distribution
deathType=deathRec['MannerOfDeath'].value_counts(sort=True)
deathCnt=deathRec['PlaceOfDeathAndDecedentsStatus'].value_counts(sort=True)

print ("\nDeath Agent code & description\n",deathBy)
print ("\n Death Type\n",deathType)
#print(deathCnt)

# Loading the race data
raceData=pd.read_csv("../input/Race.csv")

print (raceData)
print (deathRec['Race'].value_counts(sort=True))

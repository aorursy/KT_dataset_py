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

# Reading the data for Country only
countryData= pd.read_csv("../input/Country.csv", low_memory=True)
indicatorsData= pd.read_csv("../input/Indicators.csv", low_memory=True)
# Creating a copy of the original dataset as sub. All experiments will be done on sub
subCntry=countryData
subIndicatr=indicatorsData
#Determine the number of rows and columns in the dataset
print (subCntry.shape)
print (subIndicatr.shape)
# Print the column headers/headings for Country data
namesCntry=subCntry.columns.values
print ("Country data Column Headers\n")
print (namesCntry)

# Print the column headers/headings for Indicators data
namesIndicatr=subIndicatr.columns.values
print ("\nIndicators data Column Headers\n")
print (namesIndicatr)
# print the rows with missing data
print ("\nThe count of rows with missing Country data: \n", subCntry.isnull().sum())
print ("\n\nThe count of rows with missing Indicators data: \n", subIndicatr.isnull().sum())
# Set the rows with missing data as -1 using the fillna()
subCntry=subCntry.fillna(-1)
# checking the rows again for missing data
print ("The count of rows with missing data: \n", subCntry.isnull().sum())
# Show the Frequency distribution for Regions in the world
print ("\n Regions in the world")
regionData=subCntry['Region'].value_counts(sort=True,dropna=False)
print (regionData)
# Show the frequency distribution for Indicators
print ("\n Indicators\n")
indicatrData=subIndicatr['IndicatorName'].value_counts(sort=True,dropna=False)
print(indicatrData)

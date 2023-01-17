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
# Loading the dataset's
timesData= pd.read_csv("../input/timesData.csv")
shanghaiData= pd.read_csv("../input/shanghaiData.csv")
schcntryData= pd.read_csv("../input/school_and_country_table.csv")
edusupplmData=pd.read_csv("../input/educational_attainment_supplementary_data.csv")

#Determine the number of rows and columns in the dataset's
print ("Times Data = ",timesData.shape)
print ("Shanghai Data = ",shanghaiData.shape)
print ("school and country table Data = ",schcntryData.shape)
print ("educational attainment supplementary Data = ",edusupplmData.shape)
# Print the column headers/headings
names=timesData.columns.values
print ("Times Data\n",names)
print ("--------------------\n")
names=shanghaiData.columns.values
print ("Shanghai Data\n",names)
print ("--------------------\n")
names=schcntryData.columns.values
print ("School and Country Data\n",names)
print ("--------------------\n")
names=edusupplmData.columns.values
print ("Educational Attainment Supplementary Data\n",names)
print ("--------------------\n")
# print the rows with missing data
print ("The count of rows with missing values in Times data: \n", timesData.isnull().sum())
print ("The count of rows with missing values in Shanghai data: \n", shanghaiData.isnull().sum())
print ("The count of rows with missing values in School and Country data: \n", schcntryData.isnull().sum())
print ("The count of rows with missing values in Educational Attainment Supplementary data: \n", edusupplmData.isnull().sum())
# Creating a copy of the original dataset's as cpy*. All experiments will be done on the copies
cpyTimes=timesData 
cpyShanghai=shanghaiData 
cpySchCnt=schcntryData 
cpyEduSup=edusupplmData
# Set the rows with missing data as -1 using the fillna(). 
# school_country_table_data has no missing values as shown above
cpyTimes=cpyTimes.fillna(-1)
cpyShanghai=cpyShanghai.fillna(-1)
cpyEduSup=cpyEduSup.fillna(-1)

# Checking for missing values again
print (cpyTimes.isnull().sum())
print ("\n----------\n")
print (cpyShanghai.isnull().sum())
print ("\n----------\n")
print (cpySchCnt.isnull().sum())
print ("\n----------\n")
print (cpyEduSup.isnull().sum())
print ("\n----------\n")
# Show the Frequency distribution
print ("\n Schools around the world as per the school country table data")
print("\n----------------------------------------------------------------\n")
worldSchools=cpySchCnt['school_name'].value_counts(sort=True,dropna=False)
print (worldSchools)
print("\n------------------------------------------------\n")
worldSchCntry=cpySchCnt['country'].value_counts(sort=True,dropna=False)
print (worldSchCntry)
# Show the frequency distribution for the Times Data
print ("\n International student's enrolled as per the Times Data")
print("\n----------------------------------------------------------------\n")
intrStud=cpyTimes['international_students'].value_counts(sort=True,dropna=False)
print (intrStud)
print ("\n Universities World Ranking as per the Shanghai Dataset\n")
wrldRank=cpyShanghai['World Rank'].value_counts(sort=True,dropna=False)
print (wrldRank)

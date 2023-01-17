# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Reading data from googleplaystore.csv file.

data = pd.read_csv('../input/googleplaystore.csv')

data
# information about columns type and values

data.info()
# Removing null values from data

data1=data.dropna(axis=0)



# Changing type of Reviews column

data1.Reviews=data1.Reviews.astype(int)



# information about updated dataframe

data1.info()
# Calculating and printing total no. of categories of app

cat = data1["Category"].nunique()

print("No. of categories : ",cat)



# Most and least popular categories

catGroup = data1.groupby("Category")["Category"].count()

print("Most popular category and its percentage :  ",catGroup.idxmax(),"  ,  ",round((catGroup.max()  * 100)/catGroup.sum(),2),"%",sep="")

print("Least popular category and its percentage :  ",catGroup.idxmin(),"  ,  ",round((catGroup.min()  * 100)/catGroup.sum(),2),"%",sep="")



# Bar graph for analyzing categories column

catGroup.plot.bar(figsize=(30,10))
# grouping Rating column, maximum and min rating with count, plot for further analysis 

ratingGroup=data1.groupby("Rating")["Rating"].count()



print("Maximum rating and its percentage :  ",data1["Rating"].max(),"  ,  ",round((ratingGroup[data1["Rating"].max()]  * 100)/ratingGroup.sum(),3),"%",sep="")

print("Minimum rating and its percentage :  ",data1["Rating"].min(),"  ,  ",round((ratingGroup[data1["Rating"].min()]  * 100)/ratingGroup.sum(),3),"%",sep="")



ratingGroup.plot.bar(figsize=(30,10))
# creating reviews description and changing to int type for analysis

ratingDesc = data1["Reviews"].describe()

ratingDesc = ratingDesc.apply(np.int64)



# printing reviews description

print("Reviews statistics description \n",ratingDesc,sep="")



# Grouping and plotting reviews column for analysis

reviewsGroup=data1.groupby("Reviews",as_index=False)["App"].count()

reviewsGroup.plot.scatter(x='Reviews',y='App',figsize=(30,10))
# modifying data to do operations on Size coulmn

sizeModifiedData=data1[data1["Size"]!="Varies with device"]



# Maximum and minimum size

print("Maximum app size : ",sizeModifiedData["Size"].max())

print("Minimum app size : ",sizeModifiedData["Size"].min())
# Maximum and minimum Installs

print("Maximum app Installs : ",data1["Installs"].max())

print("Minimum app Installs : ",data1["Installs"].min())
# count of free and paid games

appTypeF = data1[data1.Type == "Free"].Type.count()

appTypeP = data1[data1.Type == "Paid"].Type.count()



# percentages of free and paid games

print("Percentage of free games : ",round((appTypeF * 100/(appTypeF+appTypeP)),2),"%")

print("Percentage of Paid games : ",round((appTypeP * 100/(appTypeF+appTypeP)),2),"%")
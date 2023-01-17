#loading necessary libraries to perform EDA

import numpy as np # linear algebra

import pandas as pd # data processing

import seaborn as sns #visualisations

import matplotlib.pyplot as plt #visualisations
#loading the input data as below

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#Reading the data into dataframes

facilities = pd.read_csv("/kaggle/input/mental-health-and-suicide-rates/Facilities.csv")
#rows,columns

print("Dimensions(rows,columns): ", facilities.shape)



#column datatypes

print("Columns Datatypes:") 

print(facilities.dtypes)



#top 5 rows

facilities.head()

#Let's check for null values

print(facilities.isnull().sum())
#dropping columns

facilities = facilities.drop(columns=["day _treatment","residential_facilities"])



#filling NaN in other columns with Median

facilities["Mental _hospitals"].fillna(facilities["Mental _hospitals"].median(), inplace = True)

facilities["health_units"].fillna(facilities["health_units"].median(), inplace = True)

facilities["outpatient _facilities"].fillna(facilities["outpatient _facilities"].median(), inplace = True)



#Let's check again for null values

print(facilities.isnull().sum())



#Let's look at head again

facilities.head()
#Removing year as it of no value further.

facilities = facilities.drop(columns=["Year"])
sns.boxplot(x=facilities["Mental _hospitals"])
median = facilities['Mental _hospitals'].median()

facilities["Mental _hospitals"] = np.where(facilities["Mental _hospitals"] >1, median,facilities['Mental _hospitals'])
sns.boxplot(x=facilities["Mental _hospitals"])
sns.boxplot(x=facilities["health_units"])
median = facilities['health_units'].median()

facilities["health_units"] = np.where(facilities["health_units"] >1, median,facilities['health_units'])
sns.boxplot(x=facilities["health_units"])
sns.boxplot(x=facilities["outpatient _facilities"])
median = facilities['outpatient _facilities'].median()

facilities["outpatient _facilities"] = np.where(facilities["outpatient _facilities"] >5, median,facilities['outpatient _facilities'])
sns.boxplot(x=facilities["outpatient _facilities"])
#plotting Histogram of mental hospitals

facilities["Mental _hospitals"].hist(bins=20, alpha=0.8) #pandas function
# Finding the relations between the various columns using Heat Map

plt.figure(figsize=(5,5))

corelations= facilities.corr()

sns.heatmap(corelations,cmap="BrBG",annot=True)

corelations #displaying heatmap
plt.scatter(facilities["Mental _hospitals"],facilities["health_units"])

plt.xlabel('Mental Hospitals')

plt.ylabel('Health Units')

plt.show()
plt.scatter(facilities["Mental _hospitals"],facilities["outpatient _facilities"])

plt.xlabel('Mental Hospitals')

plt.ylabel('Outpatient Facilities')

plt.show()
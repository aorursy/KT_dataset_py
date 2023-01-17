#This is my first handon kaggle data set i am begginer in Data analytics and machine learning and will try 

#to fetch as much information i can from the avilable dataset



#importing necessary set of library

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline







#import the chennai_reservoir_level data in dataframe

crl_df=pd.read_csv("../input/chennai_reservoir_levels.csv")

crl_df.head(5)
#lets find the shape of data

crl_df.shape



#It shows that given data set have 5647 rows and 5 columns
#First lets check the data type so that we can handle the missing or null values 

crl_df.dtypes


#Checking the presence of any missing values

crl_df.isna()

crl_df.head(5)



# Total missing values for each feature

print (crl_df.isnull().sum())

# Any missing values?

print (crl_df.isnull().values.any())

#we can see that two of the reservoirs have 0.0 reading on some given dates which is not possible

#so would try to replace these values with the median





print (crl_df.head(5))

#we can see that two reservoirs have 0.0 values in their reservoir, we will replace that with mean

print (crl_df['CHOLAVARAM'].mean())

print (crl_df['CHEMBARAMBAKKAM'].mean())

# we dont want to change the original dataframe, we will create a duplicate data frame 

crl_df_copy=crl_df.copy()

crl_df_copy.head(5)



#Replacing the 0 value from CHOLAVARAM reservoir with mean using Mask method

mean_CHOLAVARAM_reservoir = crl_df_copy['CHOLAVARAM'].mean(skipna=True)

print (mean_CHOLAVARAM_reservoir)



crl_df_copy['CHOLAVARAM']=crl_df_copy.CHOLAVARAM.mask(crl_df_copy.CHOLAVARAM == 0,mean_CHOLAVARAM_reservoir)

print (crl_df_copy.head(5))
#Replacing the 0 value from CHEMBARAMBAKKAM reservoir with mean using Mask method

mean_CHEMBARAMBAKKAM_reservoir = crl_df_copy['CHEMBARAMBAKKAM'].mean(skipna=True)

print (mean_CHEMBARAMBAKKAM_reservoir)



crl_df_copy['CHEMBARAMBAKKAM']=crl_df_copy.CHEMBARAMBAKKAM.mask(crl_df_copy.CHEMBARAMBAKKAM == 0,mean_CHEMBARAMBAKKAM_reservoir)

print (crl_df_copy.head(5))
#5 point summary of numerical attributes

crl_df_copy.describe()
#Distribution of water in CHOLAVARAM Reservoir

sns.distplot(crl_df_copy['CHOLAVARAM'],kde=True)

#Distribution of water in POONDI Reservoir

sns.distplot(crl_df_copy['POONDI'],kde=True)

#Distribution of water in REDHILLS Reservoir

sns.distplot(crl_df_copy['REDHILLS'],kde=True)

#Distribution of water in CHEMBARAMBAKKAM Reservoir

sns.distplot(crl_df_copy['CHEMBARAMBAKKAM'],kde=True)
#Plotting each reservoir in boxplot to see the outliers 

crl_df_copy.boxplot()
#drawing the histogram to graphically see the data

crl_df_copy.hist()
#Measure of skewness 

from scipy.stats import skew 

print (skew(crl_df_copy['POONDI'],axis=0, bias=True))

print (skew(crl_df_copy['CHOLAVARAM'],axis=0, bias=True))

print (skew(crl_df_copy['REDHILLS'],axis=0, bias=True))

print (skew(crl_df_copy['CHEMBARAMBAKKAM'],axis=0, bias=True))



#We can see that Redhills is negatively skewed that means it is more close to the normal distribution

#to others
#Using pairplot now 

sns.pairplot(crl_df_copy)
#Measure spread variance and distribution 



#Variance of the dataframe:

print (crl_df_copy.var(axis=0))



#here we can see that the poondi has the most consistent data while Chembarambakkam is having most variance 

print ('Mean of columns')

print (crl_df_copy.mean(axis=0))

print ('STD of columns')

print (crl_df_copy.std(axis=0))



print ('This mean that in reservoir poondi 68 % of time \n Water level would be either Mean+STD or Mean_STD')

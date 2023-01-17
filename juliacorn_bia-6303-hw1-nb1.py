#import packages.

%matplotlib inline 

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
#read in the data file using pandas

Auto = pd.read_csv("../input/auto-ds-v3/auto_v2.csv")

Auto
#find the data types of each attribute

Auto.dtypes
Auto.median()
#populate the null values with the median value for the feature

Auto.fillna(Auto.median())
#compute the mean(average) values for all numerical attributes in the data frame

Auto.mean()
#compute the standard deviation for all numerical attributes in the data frame

Auto.std()
#find the distribution of categorical values in the 'Drive Type' attribute

Auto['Drive Type'].value_counts()
#find the distribution of categorical values in the 'Fuel Type' attribute

Auto['Fuel Type'].value_counts()
#get the stats function from the scipy package

from scipy.stats import zscore
zscore(Auto["Weight (lbs)"])
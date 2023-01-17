# Import all the library 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Load the Data to DataFrame

df=pd.read_csv('../input/museums.csv');

print(df.describe());
# Find the data for numeric value and group.  

# As we are working for on Museums data for checking Which organization has more revenue Zoo OR Museum

# print(df.shape[1])

# revenuIndx=df.shape[1];

# data=df.iloc[:, [4,revenuIndx-1]]

# print(data.head())
print(df.info(verbose=True));
# Count the Total Number of Recrod for each type of Museum

print(df["Museum Type"].value_counts());
# Define the group

zoosMuseums=df["Museum Type"]=='ZOO, AQUARIUM, OR WILDLIFE CONSERVATION';

# All Museums Except Zoos

otherMuseums=df["Museum Type"]!='ZOO, AQUARIUM, OR WILDLIFE CONSERVATION'; 



# load the revenue only for Zoos type museums    

zooRevenue=df[zoosMuseums]["Revenue"]

print(zooRevenue.shape)

print(zooRevenue.describe())



# load the revenue for other type museums    

otherRevenue=df[otherMuseums]["Revenue"]

print(otherRevenue.shape)

print(otherRevenue.describe());

# Remove the NAN record and duplicate record form the dataset

print("Zoo type Museum Data shape before removing duplicate. {}".format(zooRevenue.shape))

zooRevenue=zooRevenue.drop_duplicates();

zooRevenue= zooRevenue.dropna();

print("Zoo type Museum  Data shape After removing duplicate. {}".format(zooRevenue.shape))



# Remove the NAN record and duplicate record form the dataset

print("Non-Zoo Data shape before removing duplicate. {}".format(otherRevenue.shape))

otherRevenue=otherRevenue.drop_duplicates();

otherRevenue= otherRevenue.dropna();

print("Non-Zoo  Data shape After removing duplicate. {}".format(otherRevenue.shape))

print(stats.ttest_ind(zooRevenue,otherRevenue,equal_var=False));
sns.distplot(zooRevenue, kde=False, rug=True);
sns.distplot(otherRevenue, kde=False, rug=True);
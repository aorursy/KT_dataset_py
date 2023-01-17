# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns  #visualization

import matplotlib.pyplot as plt #build plot as well as styling plot



pd.set_option('display.max_rows', 10)



import sklearn as sk



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
mallData = pd.read_csv("../input/Mall_Customers.csv")
mallData.head()
mallData.info()
mallData.isnull().values.any()
sns.set(style="darkgrid")       #style the plot background to become a grid

genderCount  = sns.countplot(x="Gender", data =mallData).set_title("Gender_Count")  

genderCountAge = sns.boxplot(x="Gender", y = "Age", data =mallData).set_title("Gender_Count")  
genderCountAge = sns.swarmplot(x="Gender", y = "Age", data =mallData).set_title("Gender_Count")  
mallData["Annual Income (k$)"].describe()
mallData["Annual Income (k$)"].unique()
genderCountIncome = sns.boxplot(y="Gender", x = "Annual Income (k$)", data =mallData).set_title("Gender Count by Annual Income")  
g = sns.violinplot(y ="Annual Income (k$)", x= "Age", data =mallData).set_title("Annual Income Distribution by Age")

plt.xticks(rotation='vertical')
mallData["Spending Score (1-100)"].describe()

mallData["Spending Score (1-100)"].unique()
ageDisSpend = sns.relplot(x="Age", y = "Spending Score (1-100)", data =mallData)
ageDisGender = sns.boxplot(x="Gender", y = "Spending Score (1-100)", data =mallData)
with sns.axes_style("white"):

    sns.jointplot(x="Annual Income (k$)", y = "Spending Score (1-100)", data =mallData)

with sns.axes_style("white"):

    sns.jointplot(x="Annual Income (k$)", y = "Spending Score (1-100)", kind = "kde", data =mallData)
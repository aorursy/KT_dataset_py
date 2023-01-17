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
import pandas as pd
data=pd.read_csv('../input/train_AV3.csv')
#Reading the Data from the source
data.shape
#Getting idea about the size of the data set
data.describe()
#Getting neccesary informations about the numerics of the data set
data.head()
#Getting the first 5 elements of the data set
data.mean()
#Calculating The mean values of each numerical column
data.median()
#Calculating median
data.mode().head(1)
#Displaying mode data
new_mean=data[data.columns].fillna(data[data.columns].mean(),inplace=False)
new_mean.head()
#Replacing the unavailable datas with the mean values
new_median=data[data.columns].fillna(data[data.columns].median(),inplace=False)
new_median.head()
#Replacing the unavailable datas with median
new_mode=data[data.columns].fillna(data[data.columns].mode(),inplace=False)
new_mode.head()
#Replacing the unavailable datas with mode
data.head()
#confirming that the original data set remains unchanged
import matplotlib.pyplot as plt
#tool to plot different graphs
new_mean.plot.box()
#Drawing box diagram of the data
new_median.plot.box()
new_mode.plot.box()
new_mean.plot.bar()
#drawing bar diagram of the datas
new_median.plot.bar()
new_mode.plot.bar()
new_mean.LoanAmount.plot.hist()
#drawing histogram of a particular data
new_mean.plot.hist()
new_median.plot.hist()
new_mode.plot.hist()
new_data=new_mean
#copying the new_mean data to new_data
new_data['Category']='a'
#creating a new column with a random value 
new_data.head()
new_data.Category[new_data['ApplicantIncome'] <2877.5 ]='Lower Class'
new_data.Category[(new_data['ApplicantIncome'] >2877.5) & (new_data['ApplicantIncome']<3812.5) ]='Lower Middle Class'
new_data.Category[(new_data['ApplicantIncome'] >3812.5) & (new_data['ApplicantIncome']<5795.0) ]='Upper Middle Class'
new_data.Category[(new_data['ApplicantIncome'] >5795.0) & (new_data['ApplicantIncome']<81000.0) ]='Upper Class'
#making classifications based on applicant income
new_data.head()
#ASSIGNMENT DONE 



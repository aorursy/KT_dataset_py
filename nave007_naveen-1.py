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
#Reading the csv file:
import pandas as pd
test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')
print(test.head())
print(train.head())
#Data stucture of the dataset:
test.info()
#Data stucture of the dataset:
train.info()
#Summary of the dataset:
test.describe(include='all')
#Summary of the dataset:
train.describe(include='all')
#Dimensions of the dataset:
print(test.shape)
#Dimensions of the dataset:
print(train.shape)
#Adding the missing last column in the test dataset:
test['Loan_Status']= np.nan
print(test.head())
#creating a copy of the test set:
test_meanset=test.copy(deep=True)
#Replacing the categorical variable by the mean value:
test_meanset['ApplicantIncome'].fillna(value=4805.599455 , inplace=True)
print(test_meanset.head())
#creating a copy of the train set:
train_meanset=train.copy(deep=True)
#Replacing the categorical variable by the mean value:
train_meanset['ApplicantIncome'].fillna(value=5403.459283 , inplace=True)
print(train_meanset.head())
#creating a copy of the test set:
test_medianset=test.copy(deep=True)
#Replacing the categorical variable by the median value:
test_medianset['ApplicantIncome'].fillna(value=3786.000000 , inplace=True)
print(test_medianset.head())
#creating a copy of the train set:
train_medianset=train.copy(deep=True)
#Replacing the categorical variable by the median value:
train_medianset['ApplicantIncome'].fillna(value=3812.500000 , inplace=True)
print(train_medianset.head())
#Plottinh histogram:
import matplotlib.pyplot as plt
test_meanset['ApplicantIncome'].hist(color='white', edgecolor='red')
plt.title("Mean Histogram")
plt.xlabel("X-axis")
plt.ylabel("ApplicantIncome")
plt.show()
#Plottinh histogram:
train_meanset['ApplicantIncome'].hist(color='white', edgecolor='red')
plt.title("Mean Histogram")
plt.xlabel("X-axis")
plt.ylabel("ApplicantIncome")
plt.show()
#plotting histograms:
test_medianset['ApplicantIncome'].hist(color='white', edgecolor='red')
plt.title("Median Histogram")
plt.xlabel("X-axis")
plt.ylabel("ApplicantIncome")
plt.show()
#plotting histograms:
train_medianset['ApplicantIncome'].hist(color='white', edgecolor='red')
plt.title("Median Histogram")
plt.xlabel("X-axis")
plt.ylabel("ApplicantIncome")
plt.show()

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
#import the training data
train_data=pd.read_csv("../input/application_train.csv")
train_data.head()
test_data=pd.read_csv('../input//application_test.csv') #import test data set 
test_data.head() # to show first 5 rows of test data set
#to check no of rows and columns of the train data
train_data.shape
#to check no of rows and columns of the test data
test_data.shape
#to check datatypes of columnsin training set

train_data.dtypes

train_data.describe()   #describe method is used to check quartiles of numberical columns

#Check Categorical data and describe method
categorical=train_data.dtypes[train_data.dtypes=='object'].index
print(categorical)
train_data.dtypes[train_data.dtypes=='object'].count()  #Total number of categorical variable in training set
train_data[categorical].describe() #since mathematical operations can't be performed on object data types, only count,unique,top and freqency of thae column will be  shown.
total_missed=train_data.isnull().sum().sort_values(ascending=False)
type(total_missed)
print(total_missed.head(10))
total=train_data.isnull().count().sort_values(ascending=False)
type(total)
missing_percent= (train_data.isnull().sum()/train_data.isnull().count()*100).sort_values(ascending=False)
type(missing_percent)
missing_train_data= pd.concat([total_missed,total,missing_percent],axis=1,sort=False,keys=['Total_missed','total_rows','missed_percent'])
missing_train_data.head()
train_data['TARGET'].value_counts().index
#Target is our independent variable and 0 are those who repaid the loans on time and 1 are the those who failed to pay on time
labels=train_data['TARGET'].value_counts().index
sizes=train_data['TARGET'].value_counts().values
    
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.pie(sizes,labels=labels,colors=('Blue','yellow'),autopct='%1.1f%%', shadow=True, startangle=90)
plt.legend(labels)
plt.axis('equal') #to make pie chart symmetric 
plt.title('Loan repaid vs not repaid')
plt.show()


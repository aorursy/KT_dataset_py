# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings  

warnings.filterwarnings("ignore")   # ignore warnings



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Load the data from csv file

data = pd.read_csv('../input/Mall_Customers.csv')
data.head()
# The data has 5 columns. One of them is object type and the other ones are integer.

# The data has no null value.

data.info()
# This function gives us some statistical information about our data.

data.describe()
# Correlations within the columns

data.corr()
# Male - Female distribution of data

sns.countplot(x="Gender", data=data)

data.loc[:,'Gender'].value_counts()
# we pull the column that contains categoric data

data.iloc[:,1:2]
gender = data.iloc[:,1:2].values

print(gender)
# In this step, we transform the pulled data from categoric to numeric 

# Import needed encoder from scikit-learn library

# Due to encoding, categoric data is transformed to numeric data namely 0 and 1s.



from sklearn.preprocessing import LabelEncoder



# creation an object

le = LabelEncoder() 

gender[:,0] = le.fit_transform(gender[:,0])



print(gender)

print(type(gender))
# convert result from numpy array to pandas dataframe



result = pd.DataFrame(data=gender)

print(type(result))

result.columns=['gender']

result
result2 = data.iloc[:, 2:4]

result2
print(type(result2))
# Concatenating the dataframes

result3 = pd.concat([result, result2], axis=1)

result3
score = data.iloc[:, 4:]

score
print(type(score))
# In this step, we create train and test sets from the data.

# Later, our machine will learn from x_train and y_train sets and it will predict y_test set from x_test. 



from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(result3, score, test_size=0.33, random_state=0)
x_train
x_test
y_train
y_test
# Standardization is the process of putting different variables on the same scale.

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(x_train)

X_test = sc.fit_transform(x_test)

Y_train = sc.fit_transform(y_train)

Y_test = sc.fit_transform(y_test)
X_train
X_test
Y_train
Y_test
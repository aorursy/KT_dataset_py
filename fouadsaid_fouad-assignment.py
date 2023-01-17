# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt







data = pd.read_csv("../input/loan.csv", low_memory = False)

data = data[(data.loan_status == 'Fully Paid') | (data.loan_status == 'Default')]

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory





print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#return the shape of the data

data.shape
#create a target variable called 'target' that takes the value 1 if 'loan_status' is Fully Paid and takes value 0 if 'loan_status' is 'default'



data['target']=(data['loan_status'] == 'Fully Paid')
#HW1: returns the length of the data

len(data)
#HW2

#outputing the mean of loan amount

print("The mean of loan amount:")

print(data["loan_amnt"].mean())



#outputing the median of loan amount

print("The Median of loan amount:")

print(data["loan_amnt"].median())



#outputing the maximum value of loan amount

print("The Maximum Value of loan amount:")

print(data["loan_amnt"].max())



#outputing the standard deviation of loan amount

print("The Standard Deviation of loan amount:")

print(data["loan_amnt"].std())



#HW2

#drawing a histogram of the loan amounts against the no. of loans

x = data.loan_amnt

plt.hist(x, bins=20)

plt.ylabel('No. of loans')

plt.xlabel('Amnt of loan ($)')

plt.show()
#HW3

#Creating boxplot of interest rate versus short and long term

data.boxplot('int_rate', by = 'term')
#HW4

#grouping the data by term length

term_groups = data.groupby('term')



#calculating the mean per term group

print("Mean of Term Grouped")

print(term_groups['int_rate'].mean())



#calculating the mean interest rate per term group

print("Mean of Interest rate Term Grouped")

print(term_groups['int_rate'].std())
#HW4

#grouping the data by debt grade

grade_groups = data.groupby('grade')

#calculating the mean interest rate per grade group

grade_groups['int_rate'].mean()



#HW5 

#Calculate total amount loaned for each debt grade

total_loaned = grade_groups['funded_amnt'].sum()

print(total_loaned)
#HW5

#Calculate total amount received for each debt grade

total_received = grade_groups['total_pymnt'].sum()

total_received
#HW5

#Calculate % yield for each debt grade

(total_received / total_loaned - 1) * 100
#HW6

#Counts each result for the feature application_type and groups them

data['application_type'].value_counts()
#HW6. Does it make sense to use this feature?

# No, almost all applications are individual; it can be assumed that other variables would

# prove to be more statistically significant.
#HW7 

#Convert 10 variables into dummies

X = pd.get_dummies(data[['term','verification_status','purpose','policy_code', 'loan_amnt','funded_amnt', 'funded_amnt_inv','int_rate', 'emp_length', 'addr_state']])

X.shape
#HW8

#Then need to divide the data into training and testing sets

#Setting the y variable and combining it with the dummy data from the previous code to train the model using

#random forest classifier

y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)  



X_train.shape

#HW9

#Then set hyper parameters for the Random Forest Classifier

#By convention, clf means 'Classifier'

clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
#train the classifier

clf.fit(X_train,y_train)
#test the classifier (apply it to the test data it has never seen before)

y_pred = clf.predict(X_test)
#calculate accuracy

accuracy_score(y_test, y_pred)
#HW10

#Predict repayment for all applicants, what is the accuracy now?

#first need to figure out how to get it to predict on another data set



#matrix of ones of the same size as test variable, use this to predict that every observation has repaid

y_pred = np.ones(y_test.shape)
#then test the accuracy of the model on another set of data outside that it was trained on

accuracy_score(y_test, y_pred)
#HW11

# Class count, which counts the fully paid and default items into two distinct classes

count_class_0, count_class_1 = data.target.value_counts()



# Divide by class

data_class_0 = data[data['target'] == 1]

data_class_1 = data[data['target'] == 0]
#HW11 

#uses under sampling to better balance the data by decreasing the no. of observation to the lowest of either

#classes to make the testing more fair

data_class_0_under = data_class_0.sample(count_class_1)

data_test_under = pd.concat([data_class_0_under, data_class_1], axis=0)



print('Random under-sampling:')

print(data_test_under.target.value_counts())



data_test_under.target.value_counts().plot(kind='bar', title='Count (target)');
#HW11 

#uses over sampling to better balance the data by increasing the no. of observation of the lowest of either

#classes to the highest of them to make the testing more fair

data_class_1_over = data_class_1.sample(count_class_0, replace=True)

data_test_over = pd.concat([data_class_0, data_class_1_over], axis=0)



print('Random over-sampling:')

print(data_test_over.target.value_counts())



data_test_over.target.value_counts().plot(kind='bar', title='Count (target)');
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import statsmodels.api as sm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# loads the data as a CSV file

data = pd.read_csv("../input/loan.csv", low_memory = False)
# Only keeps data for loans whose status is 'Fully Paid' or 'Default'

data = data[(data.loan_status == 'Fully Paid') | (data.loan_status == 'Default')]
# Transforms paid loans into a binary target variable

data['target'] = (data.loan_status == 'Fully Paid')
# Task 1. How many records are there?

# counts the data

data.shape
# Task 2. Plot the distribution of loan amounts (using a histogram) and compute the mean, median, maximum 

# and standard deviation.



# Plots the distribution of loans using a histogram

from matplotlib.pyplot import * 

matplotlib.pyplot.hist(data.loan_amnt)
#Calculates and prints the mean

print("Mean loan amount:" , np.mean(data.loan_amnt))
#Calculates and prints the median

print("Median loan amount:" , np.median(data.loan_amnt))
#Calculates and prints the standard deviation

print("Standard deviation loan amount:" , np.std(data.loan_amnt))
# Task 3a. What are the mean and standard deviations interest rates of short and long-term loans?



# Calculating mean of short-term loans

term_groups = data.groupby ('term')

term_groups['int_rate'].mean()
# Calculating standard deviation of short-term loans

term_groups['int_rate'].std()
# Task 3b. Creating a boxplot for mean and standard deviation

data.boxplot('int_rate', by = 'term')
# Task 4. How does debt grade influence the interest on the loan?

# What is the average interest rate on the debt grade with the highest average interest rate?



grade_groups = data.groupby ('grade')

grade_groups['int_rate'].mean()
# Printing the average of grade g

grade_groups['int_rate'].mean().max()
# Task 5. How does the default rate and interest rate differ between debt grades?

# What is the highest realized yield within any debt grade?



# Calculate total amount loaned for each debt grade

total_loaned = grade_groups['funded_amnt'].sum()
# Calculate total amount received for each debt grade

total_received = grade_groups['total_pymnt'].sum()
# Calculate % yield for each debt grade

(total_received / total_loaned - 1) * 100
# Task 6. Your boss suggests that you should use the feature application_type for your predictions.

# 6a. How many records for each application type are there?



# Counts each result for the feature application_type

data['application_type'].value_counts()

#6b. Does it make sense to use this feature?

# No, almost all applications are individual; it can be assumed that other variables would

# prove to be more statistically significant.



# Identifies the percentage of applications that are individual.

# This should help decide if this is a useful variable.

len(data[data.application_type.astype(str)=="Individual"])/len(data)*100
#7. You ultimately settle on the features 'loan_amnt', 'funded_amnt', 'funded_amnt_inv',

# 'term','int_rate', 'emp_length', 'addr_state', 'verification_status', 'purpose', 

# 'policy_code' as input variables for your predictive model. Convert the categorical variables

# into dummy variables using pandas get_dummies function

# What is the total width of your feature set now?



# Converts selected variables into dummy variables

X = pd.get_dummies(data[['term','verification_status','purpose','policy_code', 

'loan_amnt','funded_amnt', 'funded_amnt_inv','int_rate', 'emp_length', 'addr_state']])
# Counts the number of records and the new number of features (i.e. width) of the data set.

X.shape
#8. Use scikit learn’s train_test_split function to split your data. If your training set size

# is 33%, what is the shape of your X_train? Set the random_state to 42.



# Imports train_test_split and RandomForestClassifier from scikit learn

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
# Splits the data in two, part for training and part for testing. Set training set size to 33% 

# and random_state to 43, as per assignment instructions.

y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.33, random_state=42)



# Calculates the share of the X_train.

X_train.shape
#9. Use a random forest to train a predictive model with ‘target’ as outcome variable. Use the

# random forest classifier function.

# What out of sample accuracy do you achieve. Use n_estimators=100, max_depth=4 as hyper parameters, 

# and use test_size=0.33 for splitting. Remember to set the random_state=42 for both splitting and 

# training.



# Imports accuracy score function

from sklearn.metrics import accuracy_score
# Sets hyper parameters for the Random Forest Classifier

# by convention, clf means 'Classifier'

clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
# Trains the classifier

clf.fit(X_train,y_train)
# Tests the classifier (applies it to the test data it has never seen before)

y_pred = clf.predict(X_test)
# Calculates accuracy

accuracy_score(y_test, y_pred)
#10. Now change your model so that you predict repayment for all applicants. What is the accuracy now?

# Matrix of ones of the same size as test variable, use this to predict that every observation has 

#repaid

y_pred = np.ones(y_test.shape)
# Then test the accuracy of the model on another set of data outside that it was trained on

accuracy_score(y_test, y_pred)
#11. The loan data is imbalanced as the bulk of loans does get repaid. Read this notebook on imbalanced 

# data and adjust your sampling strategy accordingly (hint: use either the random over sampler or 

# random under sampler function, the others might take very long to run)



# Class count

count_class_0, count_class_1 = data.target.value_counts()



# Divide by class

data_class_0 = data[data['target'] == 1]

data_class_1 = data[data['target'] == 0]
data_class_0_under = data_class_0.sample(count_class_1)

data_test_under = pd.concat([data_class_0_under, data_class_1], axis=0)



print('Random under-sampling:')

print(data_test_under.target.value_counts())



data_test_under.target.value_counts().plot(kind='bar', title='Count (target)');
data_class_1_over = data_class_1.sample(count_class_0, replace=True)

data_test_over = pd.concat([data_class_0, data_class_1_over], axis=0)



print('Random over-sampling:')

print(data_test_over.target.value_counts())



data_test_over.target.value_counts().plot(kind='bar', title='Count (target)');
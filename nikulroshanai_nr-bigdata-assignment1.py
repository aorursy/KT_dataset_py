# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import statistics

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.metrics import accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.





# Load data as per assingment 1

data = pd.read_csv("../input/loan.csv", low_memory = False)

data  = data[(data.loan_status == 'Fully Paid') | (data.loan_status == 'Default')]

data['target'] = (data.loan_status == 'Fully Paid')

# Question 1 - How many records are there left? How many features are there?



data_count = data.count()

print(data_count)



len(data)

# Question 2 - Plotting a histogram and computing mean, median, min, max and Std Dev



# Histogram

plt.hist(data.loan_amnt, bins = 15)

plt.ylabel("Count")

plt.xlabel("Loan Ammounts")

plt.show()



#Mean

print("Mean = ", statistics.mean(data.loan_amnt))



#Median

print("Median = ", statistics.median(data.loan_amnt))



#Min

print("Min = ", min(data.loan_amnt))



#Max

print("Max = ", max(data.loan_amnt))



#Std Dev

print("Std Dev = ", data.loan_amnt.std())

# Question 3 - Understanding how many unique terms there are in the "term" field



data.term.unique() 

# Question 3 - Determining the mean and standard dev of the unique terms in the "term" field



data.groupby(['term']).int_rate.agg(['mean', 'std'])

# Question 3 - Box plot of interst rates for unique "terms"



data.boxplot('int_rate','term', figsize = (5,5), showfliers = False)

# Question 4 - Determining the mean and standard dev of the unique terms in the "grade" field



data.groupby(['grade']).int_rate.agg(['mean'])



# Question 5 - Determining the means of the default loans in various loan grades 



print("Means of Default")

x_array = ['A','B','C','D','E','F','G']

for i in range(7):

    x = data[(data['grade'] == x_array[i]) | (data['loan_status'] =='Default')]

    print(str(x_array[i]) + ":  " + str(x['int_rate'].mean()))

    

# Question 5 - Determining the means of the fully paid loans in various loan grades



print("Means of Fully Paid")

y_array = ['A','B','C','D','E','F','G']

for i in range(7):

    y = data[(data['grade'] == y_array[i]) | (data['loan_status'] =='Fully Paid')]

    print(str(y_array[i]) + ":  " + str(y['int_rate'].mean()))
# Question 6 - Determining the number of records for each unique term with in the "application type" field



print(data['application_type'].unique())



print("Joint App")

print(len(data[(data['application_type'] == 'Joint App')]))



print("Individual")

print(len(data[(data['application_type'] == 'Individual')]))



# Unclear at this point if these 2 features are useful in precidictions, will need to understand the R^2 of these 2 feature sets

# and how that correlates with the output.

# Question 7 - Determine the features that are non integer (c_array) and integer features (i_array) the understand its dimentions



noni_array = ['term','emp_length','addr_state','verification_status','purpose']

i_array = data[['loan_amnt','funded_amnt','funded_amnt_inv','term','int_rate',

                  'emp_length','addr_state','verification_status','purpose','policy_code']]

dummy = pd.get_dummies(data = i_array, columns = noni_array, sparse = True)

dummy.shape
# Question 8 - Splitting dummy data set into 4 arrays



dummy = dummy.round({'funded_amnt_inv': 0, 'int_rate': 0})

target = data[['target']]

target_dummy_data = pd.get_dummies(target)

x_train_array, x_test_array, y_train_array, y_test_array = train_test_split(dummy,target_dummy_data,test_size = 0.33, random_state = 42)



x_train_array.shape, x_test_array.shape, y_train_array.shape, y_test_array.shape

# Question 9 - Running random forest classifer on split data to generate predictive model and determining accuracy of model



model = RandomForestClassifier(n_estimators = 100,max_depth = 4,bootstrap = True, random_state = 42)



model.fit(x_train_array, y_train_array)



y_predict = model.predict(x_test_array)



print("Accuracy Level: ",metrics.accuracy_score(y_test_array, y_predict)*100)

# Question 10 - Accuracy score for predicted repayments of all applicants

y_predict = np.ones(y_test_array.shape)



print("New accuracy Level: ",metrics.accuracy_score(y_test_array,y_predict))
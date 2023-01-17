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



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



data = pd.read_csv("../input/loan.csv", low_memory = False)
#Only keep data for loans whose status is ‘Fully Paid’ or ‘Default’

data = data[(data.loan_status == 'Fully Paid') | (data.loan_status == 'Default')]
data['target'] = (data.loan_status == 'Fully Paid')
data.shape
# Q2. Output of the mean, maximum, median, standard deviation of loan amount



print(f'The mean of loan amount: {(data["loan_amnt"].mean())}')

print(f'The median of loan amount: {(data["loan_amnt"].median())}')

print(f'The maximum of loan amount: {(data["loan_amnt"].max())}')

print(f'The standard deviation of loan amount: {(data["loan_amnt"].std())}')

# Q2. Drawing a histogram of the loan amounts against the no. of loans 

x = data.loan_amnt

plt.hist(x)

plt.ylabel('No. of loans')

plt.xlabel('Amnt of loan ($)')

plt.show()
# Q3a. What is mean & sd of interest rates of short and long term loans



term_groups=data.groupby ('term')

term_groups['int_rate'].mean()

# std

term_groups['int_rate'].std()
# Q3b. Creating boxplot of interest rate versus short and long term 

data.boxplot('int_rate', by = 'term')

# Q4. Grouping by debt grade

  

grade_groups = data.groupby('grade')



#calculating the mean interest rate per grade group 

grade_groups['int_rate'].mean()

# Printing avg of grade g

grade_groups['int_rate'].mean().max()
# Q5.

#Calculate total amount loaned for each debt grade 

total_loaned = grade_groups['funded_amnt'].sum() 

print(total_loaned)
#Calculate total amount received for each debt grade 

total_received = grade_groups['total_pymnt'].sum() 
#Calculate % yield for each debt grade 

(total_received / total_loaned - 1) * 100
# Q6.

#Counts each result for the feature application_type and groups them 

data['application_type'].value_counts()



# Q6. Does it make sense to use this feature?

# No, almost all applications are individual; it can be assumed that other variables would prove to be more statistically significant.
# Q7.,

#Convert 10 variables into dummies

X = pd.get_dummies(data[['term','verification_status','purpose','policy_code', 'loan_amnt','funded_amnt', 'funded_amnt_inv','int_rate', 'emp_length' , 'addr_state']])

X.shape
# Q8.

#Then need to divide the data into training and testing sets

#Setting the y variable and combining it with the dummy data from the prev ious code to train the model using random forest classifier



y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train.shape
# Q9.

#Then set hyper parameters for the Random Forest Classifier, by convention, clf means 'Classifier'

clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
#train the classifier 

clf.fit(X_train,y_train)

#test the classifier (apply it to the test data it has never seen before) 

y_pred = clf.predict(X_test)
#calculate accuracy 

accuracy_score(y_test, y_pred)

# Q10.

#Predict repayment for all applicants, what is the accuracy now? #first need to figure out how to get it to predict on another data set 

#Matrix of ones of the same size as test variable, use this to predict tha t every observation has repaid

y_pred = np.ones(y_test.shape)

#then test the accuracy of the model on another set of data outside that i t was trained on

accuracy_score(y_test, y_pred)
# Q11.

# Class count, which counts the fully paid and default items into two dist inct classes

count_class_0, count_class_1 = data.target.value_counts()


# Divide by class

data_class_0 = data[data['target'] == 1] 

data_class_1 = data[data['target'] == 0]


# Q11.

#uses under sampling to better balance the data by decreasing the no. of o bservation to the lowest of either classes to make the testing more fair

data_class_0_under = data_class_0.sample(count_class_1)

data_test_under = pd.concat([data_class_0_under, data_class_1], axis=0)



print('Random under-sampling:') 

print(data_test_under.target.value_counts())

data_test_under.target.value_counts().plot(kind='bar', title='Count (targe t)');


# Q11.

#uses over sampling to better balance the data by increasing the no. of ob servation of the lowest of either classes to the highest of them to make the testing more fair 

data_class_1_over = data_class_1.sample(count_class_0, replace=True) 

data_test_over = pd.concat([data_class_0, data_class_1_over], axis=0)

print('Random over-sampling:') 

print(data_test_over.target.value_counts())

data_test_over.target.value_counts().plot(kind='bar', title='Count (target )');
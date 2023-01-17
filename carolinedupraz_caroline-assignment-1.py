# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



#Transform paid loans into a binary target variable as per Assignment 1 instructions





# Any results you write to the current directory are saved as output.
#Load the lending club data as per the Assignment 1 instructions

data = pd.read_csv("../input/loan.csv", low_memory = False)
data.shape
#Only include data for loan Status of Fully Paid and Default as per the Assignment 1 instructions

data = data [(data.loan_status == 'Fully Paid') | (data.loan_status == 'Default')]

data.shape
#Transform paid loans into a binary target variable as per Assignment 1 instructions

data ['target'] = (data.loan_status == 'Fully Paid')

data.target.head()

data.shape
#Question 1 - Count records

print("Question 1")

print("Number of records:", len(data))

print("Number of features:", str(data.shape[1]))
import matplotlib.pyplot as plt
#Question 2 - Plot Histogram

print("Question 2a")

plt.hist(data.loan_amnt, bins=20)
#Question 2b - Calc Mean

print("Question 2b")

print("Mean:", data.loan_amnt.mean())

print("Median:", data.loan_amnt.median())

print("Max:", data.loan_amnt.max())

print("Standard Deviation:", data.loan_amnt.std())
#Question 3 - Create new variable 'sterm' that takes value 1 if term is '36 months' and takes value 0 for all other values

data['sterm'] = (data.term == ' 36 months')
#Splitting data to create one data set with short term loans and one with long term loans



sterm = data[(data.term == ' 36 months')]

lterm = data[(data.term == ' 60 months')]

#Checking number of observations in each data set, then comparing total number of lines to original data set to make sure they're the same

print (len(sterm), len(lterm), len(sterm)+len(lterm), len(data))
#Question 3 - Print short term loans mean and standard deviation

print("Question 3a - Short term")

print("Mean: "+ str(sterm.int_rate.mean()))

print("Standard deviation: "+str(sterm.int_rate.std()))
#Question 3 - Print long term loans mean and standard deviation

print ("Question 3a - Long Term")

print("Mean: "+ str(lterm.int_rate.mean()))

print("Standard deviation: "+str(lterm.int_rate.std()))
#Question 3 - Plot a box plot of interest rate by term using pandas boxplot function

data.boxplot(column='int_rate', by='term')
#Question 4 Show averages by debt grade

data.int_rate.groupby(data.grade).mean()
#Question 4 - Find average interest rate on the debt grade with the highest average interest rate

print("Grade G has the highest average interest rate of", max(data.int_rate.groupby(data.grade).mean()))
#Question 5 - Show the highest of the average realized yield of debt grade

data['realized_yield'] = (data.total_pymnt/data.funded_amnt-1)

data.realized_yield.groupby(data.grade).mean()
#Question 5

#Look at difference between grades interest rates and default rates 

data.grade.unique()

for grade in data.grade.unique():

    print('Grade: '+grade+' Interest rate: %7.4f' %(data[data.grade==grade].int_rate.mean())+' Default rate: %8.7f' %(((1-data[data.grade==grade].target.mean())*100))+'%')

    

print('The grade with highest interest rate has the lowest default rate')
#Question 5 Continuation

#Calculate realised yield for each grade

grade_realized_yield = data.total_pymnt.groupby(data.grade).sum()/data.funded_amnt.groupby(data.grade).sum()-1



print(grade_realized_yield)



#Simply print the key with highest value together with its value

print('The grade with highest realized yield is '+grade_realized_yield.idxmax()+' with realizeed yield '+str(grade_realized_yield.max()))
#Question 6 - Find records for each application type 

data['application_type'].value_counts()
#You need to see if application types actually change something through statistical analysis (SD, mean, histograms). The type of application submitted might not predict default rate.
#Question 7 - After converting the categorical variables into dummy variables, find total width of new feature set

array = ['term','emp_length','addr_state','verification_status','purpose']

feature_subset = data[['loan_amnt','funded_amnt','funded_amnt_inv','term','int_rate','emp_length','addr_state','verification_status','purpose','policy_code']]

feature_subset.head()

dummy_data = pd.get_dummies(data = feature_subset, columns= array, sparse=True)

dummy_data.shape
#Question 8 - Import train test 

from sklearn.model_selection import train_test_split
#Split the dataset set into training and testing sets. Show shape of data.

dummy_data = dummy_data.round({'funded_amnt_inv': 0, 'int_rate': 0})

target_data = data[['target']]

target_dummy_data = pd.get_dummies(target_data)

target_data.head()

x_train, x_test, y_train, y_test = train_test_split(dummy_data,target_dummy_data,train_size=0.33, random_state=42)

x_train.shape
#9: Use scikit learn’s train_test_split function to split your data. If your training set size is 33%, what is the shape of your X_train? Set the random_state to 42.

from sklearn.model_selection import train_test_split



target = np.array(data['target']) # create a list of labels
#Split the dataset set into trainign and testing sets

X_train, X_test, y_train, y_test = train_test_split( dummy_data, target, train_size=0.33, random_state=42)



from sklearn.ensemble import RandomForestClassifier
# Begin with 100 decision trees

rf_model = RandomForestClassifier(n_estimators = 100, max_depth=4, random_state = 42)



# Train the model using training data

rf_model.fit(X_train, y_train)



#Run predictions on test data

print('Out of sample accuracy: '+str(rf_model.score(X_test, y_test)*100)+'%')
#Question 10 -Predicting for all payments. Assumption is that we are using the entire dataset, including data that was used to train even though training data is never used to learn accuracy of a model

all_repayments_pred = [1]*len(y_test)



print("Accuracy on all data repayment is: "+str(rf_model.score(X_test, all_repayments_pred)*100)+"%")
#Question 11



#Create new dataframe 'data_subset' that contains only features of interest (x, independent variables)

new_data_subset = data[['loan_amnt','funded_amnt','funded_amnt_inv','term','int_rate','emp_length','addr_state','verification_status','purpose','policy_code','target']]



#Convert all categorical variables in 'data_subset' into dummy variables using function 'pd.get_dummies', and save values in new dataframe 'dummy_data'

new_dummy_data = pd.get_dummies(data = new_data_subset)

#dummy_data.shape



new_dummy_data.head()

new_dummy_data.shape



#Create counts of both treatments of 'target'

count_class_1, count_class_0 = new_dummy_data.target.value_counts()



# Divide by class

df_class_0 = new_dummy_data[new_dummy_data['target'] == 0]

df_class_1 = new_dummy_data[new_dummy_data['target'] == 1]



#Oversampling

df_class_0_over = df_class_0.sample(count_class_1, replace=True)



#Concatenate new oversampled data

df_test_over = pd.concat([df_class_1, df_class_0_over], axis=0)



df_test_over.head()

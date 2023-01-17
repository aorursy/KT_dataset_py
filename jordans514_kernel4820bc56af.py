#Import resources



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib.ticker import StrMethodFormatter

import os

import numpy as np

from sklearn.model_selection import train_test_split



print('Complete')
#Import the dataset



data = pd.read_csv("../input/loan.csv", low_memory = False)

print('Complete')
#Create a data subset 'DS' where loan status is either fully paid or default

ds = data[(data.loan_status == 'Fully Paid') | (data.loan_status == 'Default')]



#Create a target for ds where target is a binary of whether or not the loan is fully paid

ds['target'] = (ds.loan_status == 'Fully Paid')

print('Complete')
#get the number of records in ds (just paid or defaulted loans)

print('Number of records: ', ds.shape[0])



#Get the number of columns (Features) in DS

print('Number of Columns: ', ds.shape[1])
#plot a histogram of the loan amount. Add labels, remove grid, and make black.



ds.loan_amnt.hist(bins='auto', color='#000000')



plt.xlabel('Loan Amount')

plt.ylabel('Frequency')

plt.title('Loan Amount Histogram (With Auto Bins)')

plt.grid(False)
#Show descriptive statistics of Loan Amount plus median

print('Loan Amount Descriptive Statistics:')



print('Mean' , ds.loan_amnt.mean())

print('Min:' , min(ds.loan_amnt))

print('Max:' , max(ds.loan_amnt))

print('Average:' , round(ds.loan_amnt.mean()))

print('Median:', round(ds.loan_amnt.median()))

print('Std Dev:' , round(ds.loan_amnt.std()))
#Show mean and std of interest rate by term

print(ds.groupby(['term']).int_rate.agg(['mean', 'std']))



#Box Plot of Interest Rate grouped by term

ds.boxplot(column='int_rate', by='term', showfliers=False)

print(plt.show())
#Show mean, std, and median interest rate grouped by Grade

print(ds.groupby(['grade']).int_rate.agg(['mean', 'std', 'median', 'max']))



#Box Plot of Interest Rate grouped by Grade

print(ds.boxplot(column='int_rate', by='grade', showfliers=False))

print(plt.show())

#for each grade, get the total payment and divide by how much was funded. This amount - 1 is the realized yield

(ds.groupby("grade")["total_pymnt"].sum() / ds.groupby("grade")["funded_amnt"].sum()-1)*100
#Get the count of each application_type

ds.groupby(['application_type']).size()

#I am going to create a data subset of the only the features I want to use in a model.

#This will be called model_x

#I am going to create a loop to delete all columns that I do not want

#I can then append the dummy variables and have a clean data subset for my model



keep_features = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate', 'emp_length',

            'addr_state', 'verification_status', 'purpose', 'policy_code']

delete_features = []



for x in ds.columns:

    if x not in keep_features:

    

        delete_features.append(x)

        

model_x = ds.drop(delete_features, axis=1)

model_x = pd.get_dummies(model_x, columns=["purpose", "verification_status", "term","emp_length", "addr_state"])



print('Complete')

print('Columns:', model_x.shape[1])
#I want to setup my model_y as target

model_y = ds.target

model_y = pd.get_dummies(model_y, columns=['target'])



print('Complete')
#Split the data using sklearn train_test_split

#Use test size of 0.33 and random_state of 42

#show the shape of the data



X_train, X_test, y_train, y_test = train_test_split(

     model_x, model_y, test_size=0.33, random_state=42)



print('X_train Shape:', X_train.shape)

print('y_train shape:', y_train.shape)

print('X_test shape:', X_test.shape)

print('y_test shape:', y_test.shape)

#Build a model on the train data and test it on the test data



from sklearn.ensemble import RandomForestClassifier



# Create the model with 100 trees

model = RandomForestClassifier(n_estimators=100,

                               max_depth = 4,

                               bootstrap = True)

# Fit on training data

model.fit(X_train, y_train)



y_pred = model.predict(X_test)



from sklearn import metrics



print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
#Count the number of loans in each type from the original dataset (data)

print("Number by Status Types:", data.groupby("loan_status").size())

#I want a dataset with charged off, current, and default

ds=data[(data.loan_status == "Fully Paid") | (data.loan_status == "Default") | (data.loan_status == "Charged Off")]



#Create a target for ds where target where TRUE is when the status is not Fully paid

ds['target'] = (ds.loan_status != 'Fully Paid')



print('Complete')
#Delete features I do not want in my model



keep_features = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate', 'emp_length',

            'addr_state', 'verification_status', 'purpose', 'policy_code']

delete_features = []



for x in ds.columns:

    if x not in keep_features:

    

        delete_features.append(x)

        

model_x = ds.drop(delete_features, axis=1)

model_x = pd.get_dummies(model_x, columns=["purpose", "verification_status", "term","emp_length", "addr_state"])



print('Complete')
#Run the first model on the new dataset



model.fit(model_x, ds.target)



y_pred = model.predict(model_x)



from sklearn import metrics



print("Accuracy:", metrics.accuracy_score(ds.target, y_pred))
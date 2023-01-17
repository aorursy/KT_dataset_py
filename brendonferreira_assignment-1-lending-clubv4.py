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
#Imports original loan data from Lending Club

original_data = pd.read_csv("../input/loan.csv", low_memory = False)
# Creates new data set which excludes charge off loans

data = original_data[(original_data.loan_status == 'Fully Paid')|(original_data.loan_status == 'Default')]



defaulted = original_data[(original_data.loan_status == 'Default')]



#creating a new column in the data status, Fully Paid = True, Defaulted = False

data['target'] = (data.loan_status == 'Fully Paid')
#1.   How many records are there left? How many features are there?

# Provides the length (rows) of data

len(data) 

# Returns rows and columns

data.shape
#2.   Let’s understand the data. 

    #A. Plot the distribution of loan amounts (using a histogram) 

    #B.   Compute the mean, median, maximum and standard deviation.

    

#import seaborn and modifies lib as sns (for graphing)

import seaborn as sns

#import matplot and modifies lib as plt (for graphing)

import matplotlib.pyplot as plt
#2A. Seaborne plot 

sns.distplot(data['loan_amnt'],bins = 25)
#2A. Matlab plot 

# plots histogram of loan amounts

plt.hist(data.loan_amnt,)
#2B. Prints out interger strings of Loan Amount values



print("Mean of Loan Amounts: " + str(int(data.loan_amnt.mean())))

print("Max of Loan Amounts: " + str(int(data.loan_amnt.max())))

print("Std of Loan Amounts: " + str(int(data.loan_amnt.std())))

print("Median of Loan Amounts: " + str(int(data.loan_amnt.median())))
#3.   This question explores how the term of the loan is related to the interest rate charged. 

    #a.   What are mean and standard deviations of the interest rates of short and long term loans? 



    #b.   Plot a box plot of interest rate by term using pandas boxplot function
#3A Print mean and standard deviations 

short_term_loans = data[(data['term'] == " 36 months")]

long_term_loans = data[(data['term'] == " 60 months")]



print("Mean of Short Term Loans: " + str(short_term_loans['int_rate'].mean()))

print("Standard Deviation of Short Term Loans: " + str(short_term_loans['int_rate'].std()))

print("")

print("Mean of Long Term Loans: " + str(long_term_loans['int_rate'].mean()))

print("Standard Deviation of Long Term Loans: " + str(long_term_loans['int_rate'].std()))
#3b.   Plot a box plot of interest rate by term using pandas boxplot function

# Creates a boxplot with Y axis = Interest Rates and X axis on term

data.boxplot('int_rate','term')
#3b Seaborne boxplot

ax = sns.boxplot(x= data['term'], y= data['int_rate'])

ax.set(xlabel='Loan Term', ylabel='Interest Rate')
#4.   Lending Club Provides debt grade information; here we explore how debt grade influences the interest on the loan.

    #a.   What is the average interest rate on the debt grade with the highest average interest rate? 



#Average interest rate by grade

data.groupby('grade').int_rate.mean()

#Max interest rate of any grade

data.groupby('grade').int_rate.mean().max()
#5.   Here we explore how the default rate and interest rate differ between debt grades. 

    #a.   Which debt grade has the highest realized yield? 

#Default rate equates to the inverse of target mean (as target is binary)    

(1-data.target.mean())*100

#5A. Creates a new column - realised yield. Total Payment divided by Funded amount discounted back by term 



data['ryield'] = np.where(data.term == ' 36 months', 

                          np.power(((data['total_pymnt']/data['funded_amnt'])),(1/3)), 

                          np.power(((data['total_pymnt']/data['funded_amnt'])),(1/5)))

data.ryield.groupby(data.grade).max()-1
#5A. Returns max yield 



data.ryield.groupby(data.grade).max().max()-1
#6.   Your boss suggests that you should use the feature application_type for your predictions. 

    #a.   How many records for each application type are there? 

    #b.   Does it make sense to use this feature?
#Q6A Groups and counts by application type

data.groupby('application_type').grade.count()
#6 Sets out each version as a function then divides joint app by total

joint_app = data[(data['application_type'] =='Joint App')]

individual = data[(data['application_type'] =='Individual')]

len(joint_app) / len(data)

# as Joint App only equate to <2% of application it does not appear on the face of it to be a meaningful feature to include
#7. You ultimately settle on the features 'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term','int_rate', 'emp_length', 'addr_state','verification_status', 'purpose', 'policy_code' as input variables for your predictive model. 

    #Convert the categorical variables into dummy variables using pandas get_dummies function, 

    #What is the total width of your feature set now?

    

#feature subset are categories with multiple or interger based

feature_subset = data [['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term','int_rate', 'emp_length', 'addr_state','verification_status', 'purpose', 'policy_code'

]]



# array used for non-interger based result

array = [ 'term','emp_length','addr_state', 'verification_status','purpose']



#creates dummy variables of the feature_subset

dummy_data = pd.get_dummies(feature_subset, columns=array, sparse=True)



#produces dummy data shape (width)

dummy_data.shape
#8. Use scikit learn’s train_test_split function to split your data. 

    #If your training set size is 33%, what is the shape of your X_train? Set the random_state to 42.



# import SKLEARN functions for analysis

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification

#Import scikit-learn metrics module for accuracy calculation

from sklearn import metrics
#Reduce memory constraint by rounding fund amnt and interest rate

dummy_data = dummy_data.round({'funded_amnt_inv': 0, 'int_rate': 0})



#create target dummy data to train the Y variable

target_data = data[['target']]

target_dummy_data = pd.get_dummies(target_data)
#8 Set training set with X variables = dummy data features and Y variables to target dummy. Test size equal 33% of the sample.

x_train, x_test,y_train, y_test = train_test_split(dummy_data,target_dummy_data,test_size=0.33, random_state=42)
#9. Use a random forest to train a predictive model with ‘target’ as outcome variable. 

    #Use the random forest classifier function

    #Use n_estimators=100, max_depth=4 as hyper parameters, and use test_size=0.33 for splitting. 

    #n_estimators = 100 total branches / depth = how many levels 





clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)

    

    

    #What out of sample accuracy do you achieve. 



clf.fit(x_train, y_train.values.ravel())
#9 Model Accuracy, how often is the classifier correct?

#Prediction based off training

y_pred=clf.predict(x_test)



print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#10. Now change your model so that you predict repayment for all applicants. What is the accuracy now?



#matrix of ones of the same size as test variable, use this to predict that every observation has repaid

y_pred = np.ones(y_test.shape)



#then test the accuracy of the model on another set of data outside that it was trained on

metrics.accuracy_score(y_test, y_pred)



#Q10  alternative



#Use Original_data function (aka incl. charge offs)

#feature subset are categories with multiple or interger based

feature_subset = original_data [['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term','int_rate', 'emp_length', 'addr_state','verification_status', 'purpose', 'policy_code'

]]



# array used for non-interger based result

array = [ 'term','emp_length','addr_state', 'verification_status','purpose']



#creates dummy variables of the feature_subset

o_dummy_data = pd.get_dummies(feature_subset, columns=array, sparse=True)
#Reduce memory constraint by rounding fund amnt and interest rate

o_dummy_data = o_dummy_data.round({'funded_amnt_inv': 0, 'int_rate': 0})



#create target dummy data to train the Y variable

original_data['target'] = (original_data.loan_status == 'Fully Paid')

o_target_data = original_data[['target']]

target_dummy_data = pd.get_dummies(o_target_data)
x_train, x_test,y_train, y_test = train_test_split(o_dummy_data,target_dummy_data,test_size=0.33, random_state=42)

clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)

    

    

    #What out of sample accuracy do you achieve. 



clf.fit(x_train, y_train.values.ravel())
y_pred=clf.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
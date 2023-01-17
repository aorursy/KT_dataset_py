# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# LOAD DATA



# import pandas

import pandas as pd

# import data

data = pd.read_csv("../input/loan.csv",low_memory = False)

# only select loans in 'Default' or 'Fully Paid'

data = data[(data.loan_status == 'Fully Paid')|(data.loan_status == 'Default')]

# transform into binary target variable 

data['target']=(data.loan_status == 'Fully Paid')
# Task 1 - return how many records there are left

print('There are',data.shape[0],'records and',data.shape[1],'features in the remaining dataset')
# Task 2 



import matplotlib.pyplot as plt # import histogram function

plt.hist(data.loan_amnt,bins = 20) # plot the histogram

plt.title('Distribution of loan amounts') # assign title to histogram 

plt.savefig('distribution.png')
import statistics #load statistics functions 
print('Mean is:',statistics.mean(data.loan_amnt)) #calculate mean
print('Maximum is:', max (data.loan_amnt)) #calculate max
print('Median is:', statistics.median(data.loan_amnt)) #calculate median
print('Standard Deviation is:', statistics.pstdev(data.loan_amnt)) #calculate std
# Task 3 - create boxplot of interest rates, grouped by term 

data.boxplot(column = ['int_rate'],by = 'term')
print(data.groupby(by= 'term')['int_rate'].mean()) #group the data by term & return mean for both groups 
print(data.groupby(by= 'term')['int_rate'].std()) #group the data by term & return std for both groups 
data.groupby(by='grade')['int_rate'].mean() #return mean interest rate for each debt grade
# Task 4 - calculate mean interest rates by group, then select highest interest rate

print('The highest interest rate is',max(data.groupby(by='grade')['int_rate'].mean()),'%') 
# Task 5 - calculate default rates by debt grade 

(1-data.groupby(by='grade')['target'].mean())*100
data.groupby(by='grade')['loan_amnt'].sum() # calculate amount invested 
data.groupby(by='grade')['funded_amnt'].sum() # calculate amount invested 
data.groupby(by='grade')['total_pymnt'].sum() # calculate total payments
(data.groupby(by='grade')['total_pymnt'].sum()/data.groupby(by='grade')['funded_amnt'].sum())-1 # calculate realized yield 
max((data.groupby(by='grade')['total_pymnt'].sum()/data.groupby(by='grade')['funded_amnt'].sum())-1)
# Task 6 

print('The number of records in the individual group is',data.groupby(by='application_type').size()[0],'and the number of recrods in the Joint Application group is',data.groupby(by='application_type').size()[1]) #create subgroups by application type and return number of records 
# Task 7 

# seperate predictive features 

x = data[['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term','int_rate', 'emp_length', 'addr_state','verification_status', 'purpose', 'policy_code' ]]
x = pd.get_dummies(data=x,columns=['term','emp_length','addr_state','verification_status','purpose','policy_code']) # convert categorical variables into dummies 
print('After creating dummy variables the datset is', x.shape[1]) #return the width of the newly created dataframe 
Y = data['target'] # define dependent variable 
# Task 8 

from sklearn.model_selection import train_test_split # import train_test_split 
# split the dataset into xTrain and xTest

xTrain, xTest, yTrain, yTest = train_test_split(x, Y, test_size = 0.33, random_state = 42) 
print('xTrain has the shape of', xTrain.shape[0],'records and', xTrain.shape[1],'features') # return the shape of xTrain
# Task 9 

from sklearn.ensemble import RandomForestClassifier # import Random Forest Classifier 

clf = RandomForestClassifier(n_jobs=-1,n_estimators=100, max_depth=4) # define Gaussian Classifier 
clf.fit(xTrain, yTrain) # train the model on training data
yPred=clf.predict(xTest) #predict values 
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

print("Accuracy:",metrics.accuracy_score(yTest, yPred)) # return model accuracy
# Task 10 

y10 = np.ones(len(yPred)) #create vector of ones, meaning all are repaid 

print("Accuracy:",metrics.accuracy_score(yTest, y10)) # return model accuracy
# BONUS 

#create confusion matrix 

from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt



conf_mat = confusion_matrix(y_true=yTest, y_pred=yPred)

print('Confusion matrix:\n', conf_mat)
# create dataset for bonus, includes dependent and independent variables 

bonus_data =data[['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term','int_rate', 'emp_length', 'addr_state','verification_status', 'purpose', 'policy_code', 'target' ]]
count_class_0, count_class_1 = bonus_data.target.value_counts() # count classes
# Divide by class

df_class_0 = bonus_data[bonus_data['target'] == True]

df_class_1 = bonus_data[bonus_data['target'] == False]
df_class_1_over = df_class_1.sample(count_class_0, replace=True) # increase number of default observations

df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0) # merge default & paid data

print('Random over-sampling:') # print hearder

print(df_test_over.target.value_counts()) # print values of counted data



df_test_over.target.value_counts().plot(kind='bar', title='Count (target)'); #plot number of both observations against each other 
indepB = df_test_over[['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term','int_rate', 'emp_length', 'addr_state','verification_status', 'purpose', 'policy_code']]

YB = df_test_over['target'] # define dependent variables 

xB = pd.get_dummies(data=indepB,columns=['term','emp_length','addr_state','verification_status','purpose','policy_code']) # convert categorical variables into dummies 

xTrainB, xTestB, yTrainB, yTestB = train_test_split(xB, YB, test_size = 0.33, random_state = 42) #split data into train & test 

clfB = RandomForestClassifier(n_estimators=100, max_depth=4) # define Gaussian Classifier 

clfB.fit(xTrainB, yTrainB) # train the model on training data

yPredB=clfB.predict(xTestB) #predict values 

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

print("Accuracy:",metrics.accuracy_score(yTestB, yPredB)) # return model accuracy
#create new confusion matrix to check results

conf_mat = confusion_matrix(y_true=yTestB, y_pred=yPredB) 

print('Confusion matrix:\n', conf_mat)
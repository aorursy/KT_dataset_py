# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # import plotting function, needed for histogram
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Load the data and select fully paid and default loans
data = pd.read_csv("../input/lending-club-loan-data/loan.csv", low_memory = False)
data = data[(data.loan_status == 'Fully Paid') | (data.loan_status == 'Default')]
data['target'] = (data.loan_status == 'Fully Paid')
# Number of records
print(f'Number of records is {data.shape[0]}')

# Number of features
print(f'Number of features is {data.shape[1]}')
# Imports
import matplotlib.pyplot as plt

# Plot histogram
data.loan_amnt.plot.hist(title="Distribution of loan amount", bins=20)
# All at once
data.loan_amnt.describe().astype(int) #last part asks for integers
# Or one at a time
print(f'Mean is {data.loan_amnt.mean():.02f}')
print(f'Median is {data.loan_amnt.median():.02f}')
print(f'Maximum is {data.loan_amnt.max():.02f}')
print(f'Standard deviation is {data.loan_amnt.std():.02f}')
# Mean
data.groupby('term').int_rate.mean()
# Standard deviation
data.groupby('term').int_rate.std()
# Make boxplot of int_rate by term
data.boxplot(column=['int_rate'], by = 'term', figsize=(12,8))
# Look at all means by grade
for grade in data.grade.unique():
    print(f'{grade} grade has average int_rate of {data[data.grade==grade].int_rate.mean():.2f}%')
# Or just find highest avg intrest rate
data.groupby('grade').int_rate.mean().max()
# Look at differences in interest and default rates by grade
for grade in data.grade.unique():
    print(f'{grade} loans have an average interest rate of {data[data.grade==grade].int_rate.mean():.2f} and a default rate of {(1-data[data.grade==grade].target.mean())*100:.4f}')
# Realized yield by grade
data.groupby('grade').total_pymnt.sum()/data.groupby('grade').funded_amnt.sum()-1
# Which labels does feature have?
data.application_type.unique()
# Number of joint applications
sum(data.application_type == 'Joint App') #True/False is interpreted as 1/0, so you can sum over the boolean statements
# Number of individual applications
sum(data.application_type == 'Individual')
# Define features to use
numerical_features = ['loan_amnt','funded_amnt','funded_amnt_inv','int_rate', 'policy_code']
categorical_features = ['term', 'emp_length','addr_state','verification_status','purpose']

# Encode categorical features to dummies
dummies = pd.get_dummies(data[categorical_features])

# Merged dataframe with all predictors
X = pd.concat([data[numerical_features], dummies], axis=1) # axis=1 means merge along columns

# Define y
y = data.target
X.shape[1]
# Imports
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# Shape of X_train
X_train.shape
# Imports
from sklearn.ensemble import RandomForestClassifier

# Define classifier
rfc = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)

# Train classifier
rfc.fit(X_train,y_train)
# Get overview of hyper-parameters 
# (Just for understanding whats going on. these are the default settings of the model, which can all be changed if you want to do so)
RandomForestClassifier().get_params()
# Test accuracy score
rfc.score(X_test,y_test) #accuracy_score is the default score of RandomForestClassifier()
# Look at number of default loans versus paid loans in training data
print(f' # of paid loans: {sum(y_train == True)} \n \n# of default loans: {sum(y_train == False)}')
# Accuracy score when all predictions are 'Fully paid'
y_test_new = np.ones(y_test.shape[0])
rfc.score(X_test,y_test_new)
# Imports
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Define our predicted values
y_pred= rfc.predict(X_test)

# Look at confusion matrix for current test data
conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)

print('Confusion matrix:\n', conf_mat)
# Extract the data we want to use

bonus_data =data[['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term','int_rate', 'emp_length', 'addr_state','verification_status', 'purpose', 'policy_code', 'target' ]]
count_class_0, count_class_1 = bonus_data.target.value_counts() # count classes

# Separate data into observations for default loans versus fully paid loans
df_class_0 = bonus_data[bonus_data['target'] == True] # only choose observations for fully paid
df_class_1 = bonus_data[bonus_data['target'] == False] # only choose observations for default
df_class_1_over = df_class_1.sample(count_class_0, replace=True) # increase number of default observations
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0) # merge default & paid data
print('Random over-sampling:') # print hearder
print(df_test_over.target.value_counts()) # print values of counted data

df_test_over.target.value_counts().plot(kind='bar', title='Count (target)'); #plot number of both observations against each other 
indepB = df_test_over[['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term','int_rate', 'emp_length', 'addr_state','verification_status', 'purpose', 'policy_code']]
YB = df_test_over['target'] # define dependent variables
xB = pd.get_dummies(data=indepB,columns=['term','emp_length','addr_state','verification_status','purpose','policy_code']) # convert categorical variables into dummies
xTrainB, xTestB, yTrainB, yTestB = train_test_split(xB, YB, test_size =0.33, random_state = 42) #split data into train & test
clfB = RandomForestClassifier(n_estimators=100, max_depth=4) # define Gaussian Classifier
clfB.fit(xTrainB, yTrainB) # train the model on training data
yPredB=clfB.predict(xTestB) #predict values

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
print("Accuracy:", metrics.accuracy_score(yTestB, yPredB)) # return model accuracy
#create new confusion matrix to check results
conf_mat = confusion_matrix(y_true=yTestB, y_pred=yPredB)
print('Confusion matrix:\n', conf_mat)
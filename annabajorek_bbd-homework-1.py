# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing the data
import pandas as pd
data = pd.read_csv("../input/lending-club-loan-data/loan.csv", low_memory = False)
# Transforming the data
data=data[(data.loan_status=='Fully Paid')|(data.loan_status=='Default')]
data['target']=(data.loan_status == 'Fully Paid')
data.head(10)
print(data['target'])
# Question 1
print('Number of records =', len(data.axes[0])) # this gives the number of records
print('Number of features =', len(data.axes[1])) # this gives the number of features
# Question 2
import matplotlib.pyplot as plt
loan_amnt = data.loan_amnt #defining the loan amount variable
histogram = plt.hist(loan_amnt, bins=30) #plotting the histogram with loan amounts
plt.show() #getting the graphical form of the histogram
# calculating the attributes of the loan amount variable
print('mean =', loan_amnt.mean()) #mean
print('median =', loan_amnt.median()) #median
print('maximum =', loan_amnt.max()) #maximum
print('standard deviation =', loan_amnt.std()) #standard deviation
# Question 3
import numpy as np
grouped = data.groupby('term') # grouping the interest rates by term
print(grouped['int_rate'].agg([np.mean, np.std])) # calculating the mean and standard deviation of interest rates
boxplot = data.boxplot(column=['int_rate'], by=['term']) # creating the box plot for interest rates
# Question 4
avg_int_rate = data.groupby('grade')['int_rate'].mean() #getting the average interest rates for different subgrades
avg_int_rate_g5 = avg_int_rate.max() # calculating the max function to get the average in the grade with highest interest rates, that is G5
print('average interest rate in G5 grade = ', avg_int_rate_g5)
# Question 5 
print('The default rate equals = ', (1-data.target.mean())*100) #calculating the default rate 
print('The interest rate by grade:', data.groupby(by='grade').int_rate.mean()) #showing the interest rate by grade
print('The default rate by grade:', (1-data.groupby(by='grade').target.mean())*100) #showing the default rate by grade
realized_yield = data['realized_yield']=(data.groupby('grade')['total_pymnt'].sum()/ data.groupby('grade')['funded_amnt'].sum()-1) #calculating the realized yield for the data grouped by grade
print('Debt grade', realized_yield.idxmax(), 'has a realized yield of', realized_yield.max()) #finding which grade has the highest realized yield and the max realized yield value
# Question 6
groupped_app = data.groupby('application_type') #grouping the data by application type
print(groupped_app.size()) #getting the numbers of different types of applications
# Question 7
#the categorical variables chosen are the variables that can be classified using more than 1, but a finite number of categories
X = pd.get_dummies(data[['loan_amnt','funded_amnt','funded_amnt_inv','term','int_rate','emp_length','addr_state','verification_status','purpose','policy_code']])
print('Number of features =', len(X.axes[1])) # this gives the number of features in the created model
# Question 8
from sklearn.model_selection import train_test_split
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) #dividing all the data into train and test set
print('The shape of the training data is', X_train.shape) #obtaining the shape of the train data
# Question 9
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score 
clf = RandomForestClassifier(n_estimators=100, max_depth=4,random_state=42) #creating the classifier function
clf.fit(X_train,y_train) #training the model using the training dataset
y_pred = clf.predict(X_test) #finding the values predicted by the model
print(f'Random Forest Accuracy {accuracy_score(y_test,y_pred)*100:.2f} ') #calculating the model prediction accuracy
# Question 10
y_pred = np.ones(y_test.shape) #changing the predicted variables to a vector of ones
print(f'All Repayment Accuracy {accuracy_score(y_test,y_pred)*100:.2f} ') #calculating the prediction accuracy of the new model
# Bonus 
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler()
X_ros, y_ros = ros.fit_sample(X, y)
print(X_ros.shape[0] - X.shape[0], 'new random picked points')
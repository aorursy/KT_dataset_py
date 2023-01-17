# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Importing all the required libraries
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#Loading the train and test dataset
train_data = pd.read_csv("../input/bda-2019-ml-test/Train_Mask.csv")
test_data = pd.read_csv("../input/bda-2019-ml-test/Test_Mask_Dataset.csv")
#Size of the dataset 
print ("Train data shape:", train_data.shape)
print ("Test data shape:", test_data.shape)
#Viewing the first 5 rows of the train data
train_data.head()
#Viewing the first 5 rows of the test data
test_data.head()
#Information of test and train data
test_data.info()
print("--------------------------------------------------")
train_data.info()
#Collecting more information on train data
train_data.describe()
#Checking the missing value in train dataset columns
train_data.isnull().sum()
#Checking the missing value in test dataset columns
test_data.isnull().sum()
#Counting number of times flag is 1(chain is tensed) and flag is 0(chain is loose)
train_data['flag'].value_counts()
#Visualizing the number of times the flag is 0 and 1
sns.countplot(x='flag', data=train_data, palette='hls')
plt.show()
#Splitting the train into X (Independent variable) and Y (Dependent Variable) beacuse here flag is a dependent variable.
X = train_data.drop('flag', axis=1)
y = train_data['flag']
#Splitting the train data into 70% as training data and 30% as testing data to check the accuracy of the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.32, random_state=100)
#Implementing RandomForestClassifier with optimized parameters
# Choose the type of classifier. 
clf = RandomForestClassifier()

# Choose some parameter combinations to try
parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Fit the best algorithm(model) to the data. 
clf.fit(X_train, y_train)
#Predicting the output for train data
predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))
#Evaluating the model
print(confusion_matrix(y_test,predictions.round()))
print(classification_report(y_test,predictions.round()))
print(accuracy_score(y_test, predictions.round()))
#Using the model now to make predictions on test data set
y_pred = clf.predict(test_data)
# Submission of the result
Sample_Submission = pd.read_csv("../input/bda-2019-ml-test/Sample Submission.csv")
Sample_Submission['flag'] = y_pred
Sample_Submission.to_csv("../input/bda-2019-ml-test/Sample Submission.csv",index=False)
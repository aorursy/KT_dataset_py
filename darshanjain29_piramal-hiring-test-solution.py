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
train_df = pd.read_csv("/kaggle/input/piramal-data-science-hiring/Train.csv")

test_df = pd.read_csv("/kaggle/input/piramal-data-science-hiring/Test.csv")
train_df.info()
train_df.isnull().sum()
test_df.isnull().sum()
train_df['Street Type'].value_counts()
#Street Type null fix

data = [train_df, test_df]



for dataset in data:

    mean = train_df["Street Type"].mean()

    dataset['Street Type'] = dataset['Street Type'].fillna(mean)

    dataset["Street Type"] = dataset["Street Type"].astype(float)

train_df["Street Type"].isnull().sum()
train_df['Date of Creation'] = train_df['Date of Creation'].apply(pd.to_datetime)

test_df['Date of Creation'] = test_df['Date of Creation'].apply(pd.to_datetime)



train_df['Estimated Date of Completion'] = train_df['Estimated Date of Completion'].apply(pd.to_datetime)

test_df['Estimated Date of Completion'] = test_df['Estimated Date of Completion'].apply(pd.to_datetime)



train_df['Actual Date of Completion'] = train_df['Actual Date of Completion'].apply(pd.to_datetime)

test_df['Actual Date of Completion'] = test_df['Actual Date of Completion'].apply(pd.to_datetime)
train_df['CreationDate_Month'] = train_df['Date of Creation'].dt.month

train_df['CreationDate_Week'] = train_df['Date of Creation'].dt.week

train_df['CreationDate_Day'] = train_df['Date of Creation'].dt.day  



test_df['CreationDate_Month'] = test_df['Date of Creation'].dt.month

test_df['CreationDate_Week'] = test_df['Date of Creation'].dt.week

test_df['CreationDate_Day'] = test_df['Date of Creation'].dt.day



#----------#

train_df['EstimatedCompletionDate_Month'] = train_df['Estimated Date of Completion'].dt.month

train_df['EstimatedCompletionDate_Week'] = train_df['Estimated Date of Completion'].dt.week

train_df['EstimatedCompletionDate_Day'] = train_df['Estimated Date of Completion'].dt.day  



test_df['EstimatedCompletionDate_Month'] = test_df['Estimated Date of Completion'].dt.month

test_df['EstimatedCompletionDate_Week'] = test_df['Estimated Date of Completion'].dt.week

test_df['EstimatedCompletionDate_Day'] = test_df['Estimated Date of Completion'].dt.day



#----------#

train_df['ActualCompletionDate_Month'] = train_df['Actual Date of Completion'].dt.month

train_df['ActualCompletionDate_Week'] = train_df['Actual Date of Completion'].dt.week

train_df['ActualCompletionDate_Day'] = train_df['Actual Date of Completion'].dt.day  



test_df['ActualCompletionDate_Month'] = test_df['Actual Date of Completion'].dt.month

test_df['ActualCompletionDate_Week'] = test_df['Actual Date of Completion'].dt.week

test_df['ActualCompletionDate_Day'] = test_df['Actual Date of Completion'].dt.day
#Street Type null fix

data = [train_df, test_df]



for dataset in data:

    dataset['EstimatedCompletionDate_Day'] = dataset['EstimatedCompletionDate_Day'].fillna(dataset['EstimatedCompletionDate_Day'].mode()[0])

    dataset['EstimatedCompletionDate_Day'] = dataset['EstimatedCompletionDate_Day'].astype(int)

    

    dataset['EstimatedCompletionDate_Week'] = dataset['EstimatedCompletionDate_Week'].fillna(dataset['EstimatedCompletionDate_Week'].mode()[0])

    dataset['EstimatedCompletionDate_Week'] = dataset['EstimatedCompletionDate_Week'].astype(int)

    

    dataset['EstimatedCompletionDate_Month'] = dataset['EstimatedCompletionDate_Month'].fillna(dataset['EstimatedCompletionDate_Month'].mode()[0])

    dataset['EstimatedCompletionDate_Month'] = dataset['EstimatedCompletionDate_Month'].astype(int)

    

    #------#

    dataset['ActualCompletionDate_Day'] = dataset['ActualCompletionDate_Day'].fillna(dataset['ActualCompletionDate_Day'].mode()[0])

    dataset['ActualCompletionDate_Day'] = dataset['ActualCompletionDate_Day'].astype(int)

    

    dataset['ActualCompletionDate_Week'] = dataset['ActualCompletionDate_Week'].fillna(dataset['ActualCompletionDate_Week'].mode()[0])

    dataset['ActualCompletionDate_Week'] = dataset['ActualCompletionDate_Week'].astype(int)

    

    dataset['ActualCompletionDate_Month'] = dataset['ActualCompletionDate_Month'].fillna(dataset['ActualCompletionDate_Month'].mode()[0])

    dataset['ActualCompletionDate_Month'] = dataset['ActualCompletionDate_Month'].astype(int)

    

train_df["ActualCompletionDate_Week"].isnull().sum()
train_df.drop(['Actual Date of Completion', 'Estimated Date of Completion', 'Date of Creation'], inplace = True, axis = 1)

test_df.drop(['Actual Date of Completion', 'Estimated Date of Completion', 'Date of Creation'], inplace = True, axis = 1)
test_df.info()
train_df['Problem Category'].value_counts()
X_train = train_df.drop(['Problem Category', 'L_Id'], axis = 1)

Y_train = train_df['Problem Category']

X_test = test_df.drop(['L_Id'], axis=1)
from sklearn.model_selection import cross_val_score

import warnings

warnings.filterwarnings('ignore')
# 1. Naive Bayes - Gaussian



from sklearn.naive_bayes import GaussianNB



clf_gnb = GaussianNB()



clf_gnb.fit(X_train, Y_train)



Y_pred_rf  = clf_gnb.predict(X_test)



scores_rf = cross_val_score(clf_gnb, X_train, Y_train, cv = 10, scoring = "accuracy")

print ("Scores: ",scores_rf.mean()*100)
# 2. Logistic Regression



from sklearn.linear_model import LogisticRegression



clf = LogisticRegression() 

clf.fit(X_train, Y_train)



Y_pred  = clf.predict(X_test)



scores = cross_val_score(clf, X_train, Y_train, cv = 10, scoring = "accuracy")



#clf.score(X_train, Y_train)

#acc_logistic_reg = round(clf.score(X_train, Y_train)*100, 2)



print ("Scores: ",scores)

print ("Mean: ", scores.mean()*100)

print ("Standard Deviation: ", scores.std())
# 3. SVM



from sklearn import svm

clf_svm = svm.SVC()

clf_svm.fit(X_train, Y_train)



Y_pred_svm  = clf_svm.predict(X_test)



scores_svm = cross_val_score(clf_svm, X_train, Y_train, cv = 10, scoring = "accuracy")

print ("Scores: ",scores_svm.mean()*100)
# 4. Random forest



from sklearn.ensemble import RandomForestClassifier



clf_rf = RandomForestClassifier()



clf_rf.fit(X_train, Y_train)



Y_pred_rf  = clf_rf.predict(X_test)



scores_rf = cross_val_score(clf_rf, X_train, Y_train, cv = 10, scoring = "accuracy")

print ("Scores: ",scores_rf.mean()*100)
import xgboost as xgb

from xgboost import XGBClassifier



clf_xgb = XGBClassifier().fit(X_train, Y_train)



Y_pred  = clf_xgb.predict(X_test)



clf_xgb.score(X_train, Y_train)



scores_rf = cross_val_score(clf_xgb, X_train, Y_train, cv = 10, scoring = "accuracy")

print ("Scores: ",scores_rf.mean()*100)
output = pd.DataFrame({'L_Id': test_df.L_Id, 'Problem Category': Y_pred})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
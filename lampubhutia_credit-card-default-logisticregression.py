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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats, integrate

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.linear_model import LogisticRegression

%matplotlib inline

pd.options.display.float_format = '{:.2f}'.format

plt.rcParams['figure.figsize'] = (8, 6)

plt.rcParams['font.size'] = 14

cred_df=pd.read_csv("../input/attachment_default.csv")

import re

cred_df.head(10)
cred_df.head()
cred_df.info()
cred_df.info()
sns.boxplot(x='default', y='income', data=cred_df)

plt.show()
sns.lmplot(x='balance', y='income', hue = 'default', data=cred_df, aspect=1.5, fit_reg = False)

plt.show()
pd.crosstab(cred_df['default'], cred_df['student'], rownames=['Default'], colnames=['Student'])
# Convert Categorical to Numerical

default_dummies = pd.get_dummies(cred_df.default, prefix='default')

default_dummies.drop(default_dummies.columns[0], axis=1, inplace=True)

cred_df = pd.concat([cred_df, default_dummies], axis=1)

cred_df.head()

#default_dummies
# Building Linear Regression Model

from sklearn.linear_model import LinearRegression

    

X = cred_df[['balance']]

y = cred_df['default_Yes']



linreg = LinearRegression()

linreg.fit(X, y)



print(linreg.coef_)

print(linreg.intercept_)
sns.lmplot(x='balance', y='default_Yes', data=cred_df, aspect=1.5, fit_reg = True)
#calling logistic regression  ( fitting all the data for demonstration purpose. The training & test data is excuted after this.)



from sklearn.linear_model import LogisticRegression

from sklearn import metrics

logreg = LogisticRegression()

logreg.fit(X, y)

print(logreg.coef_)

print(logreg.intercept_)



y_pred = logreg.predict_proba(X)

plt.scatter(X.values, y_pred[:,1])

#plt.scatter(X.values, y)

plt.show()
# probability of  (class 0 , class 1)



y_pred
# probability of class 0 only.



y_pred[:,0]
cred_df.head()
X.head()
#splitting the data into train and test with 70:30 ratio

from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.30, random_state=13)
#calling logistic regression

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

logreg = LogisticRegression(class_weight='balanced')

logreg.fit(X, y)

print(logreg.coef_)

print(logreg.intercept_)

#fitting the model with x and y attributes of train data

#in this it is goin to learn the pattern

logreg.fit(xTrain, yTrain)
#now applying our learnt model on test and also on train data

y_log_pred_test = logreg.predict(xTest)

y_log_pred_train = logreg.predict(xTrain)
y_log_pred_test.shape
y_log_pred_train.shape
y_log_pred_test

#creating a confusion matrix to understand the classification

conf = metrics.confusion_matrix(yTest, y_log_pred_test)

conf
# save confusion matrix and slice into four pieces

confusion = metrics.confusion_matrix(yTest, y_log_pred_test)

print(confusion)

#[row, column]

TP = confusion[1, 1]

TN = confusion[0, 0]

FP = confusion[0, 1]

FN = confusion[1, 0]

print ("TP",TP)

print ("TN",TN)

print("FN",FN)

print ("FP",FP)
cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)

sns.heatmap(conf,cmap = cmap,xticklabels=['predicted_default_yes=0','predicted_default_yes=1'],yticklabels=['actual_default_yes=0','actual_default_yes=1'],annot=True, fmt="d")
# print the first 25 true and predicted responses

print('True', yTest.values[0:15])

print('Pred', y_log_pred_test[0:15])
#comparing the metrics of predicted lebel and real label of test data

print('Accuracy_Score:', metrics.accuracy_score(yTest, y_log_pred_test))
 # Method to calculate Classification Error

    



print('Classification Error:',1 - metrics.accuracy_score(yTest, y_log_pred_test))
# Method to calculate Sensitivity



print('Sensitivity or Recall:', metrics.recall_score(yTest, y_log_pred_test))

specificity = TN / (TN + FP)



print(specificity)
from sklearn.metrics import classification_report

print(classification_report(yTest, y_log_pred_test))
cred_df.head()
#Defining a sample data to test the model

# As we discussed earlier, income has no significance in default. So only balance is considered as input & X = cred_df[['balance']]



feature_cols = ['balance']

data =[817.18]

studentid_2=pd.DataFrame([data],columns=feature_cols)

studentid_2.head()
predictions_default=logreg.predict(studentid_2)

print(predictions_default)
# print the first 10 predicted responses

# 1D array (vector) of binary values (0, 1)

logreg.predict(xTest)[0:10]
# print the first 10 predicted probabilities of class membership

logreg.predict_proba(xTest)[0:10]
# print the first 10 predicted probabilities for class 1   ( predicting diabetic cases =1)

logreg.predict_proba(xTest)[0:10, 1]
# store the predicted probabilities for class 1

y_pred_prob = logreg.predict_proba(xTest)[:, 1]
y_pred_prob[0:10]
# Plotting predicion through histogram of predicted probabilities

%matplotlib inline

import matplotlib.pyplot as plt



# 8 bins

plt.hist(y_pred_prob, bins=8)



# x-axis limit from 0 to 1

plt.xlim(0,1)

plt.title('Histogram of predicted probabilities')

plt.xlabel('Predicted probability of default')

plt.ylabel('Frequency')
# predict diabetes if the predicted probability is greater than 0.1

from sklearn.preprocessing import binarize

from sklearn.preprocessing import Binarizer

from sklearn import preprocessing

# it will return 1 for all values above 0.1 and 0 otherwise

# results are 2D so we slice out the first column



y_pred = binarize(y_pred_prob.reshape(-1,1), 0.1) 
y_pred.shape
# probability with revised threshold =0.1



y_pred_prob[0:10]
# Outcome with revised threshold =0.3



y_pred[0:10]

# previous confusion matrix (default threshold of 0.5)

print(confusion)
 #The new confusion matrix (threshold of 0.1)

    

print(metrics.confusion_matrix(yTest, y_pred))
# sensitivity has increased (used to be 0.81)

print (106 / float(3 + 106))
 # specificity has decreased (used to be 0.86)

print(1812 / float(1812 + 1079))
# IMPORTANT: first argument is true values, second argument is predicted probabilities



# we pass y_test and y_pred_prob

# we do not use y_pred, because it will give incorrect results without generating an error

# roc_curve returns 3 objects fpr, tpr, thresholds

# fpr: false positive rate

# tpr: true positive rate

fpr, tpr, thresholds = metrics.roc_curve(yTest, y_pred_prob)



plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC curve for default classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)
# IMPORTANT: first argument is true values, second argument is predicted probabilities



print(metrics.roc_auc_score(yTest, y_pred_prob))
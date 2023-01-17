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
import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats, integrate

from sklearn import metrics

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



%matplotlib inline

pd.options.display.float_format = '{:.2f}'.format

plt.rcParams['figure.figsize'] = (8, 6)

plt.rcParams['font.size'] = 14
pima=pd.read_csv("../input/diabetes.csv")

pima.head()
# duplicate copy of file

pima_data=pima.copy()
print(pima_data.shape)
pima_data.corr()
# Diabetic (0) & Non Diabetic (1) cases

sns.countplot(x='Outcome',data=pima_data)

plt.show()
pima_data.groupby("Outcome").size()
pima_data.hist(bins=30, figsize=(20, 15))

plt.show()
# Histogram visulization for Diabetic cases only  



diab1=pima_data[pima_data['Outcome']==1]

diab1.hist(bins=30, figsize=(20, 15))

plt.show()
pima_data.describe()
pima_data.plot(kind= 'box' , subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(15,10))

plt.show()
# To find missing or null data points. 



pima_data.isnull().sum()

pima_data.isna().sum()
print("Count_BP_0 : ", pima_data[pima_data.BloodPressure == 0].shape[0])



print(pima_data[pima_data.BloodPressure == 0].groupby('Outcome')['Age'].count())
print("Count_Glucose_0 : ", pima_data[pima_data.Glucose == 0].shape[0])



print(pima_data[pima_data.Glucose == 0].groupby('Outcome')['Age'].count())
print("Count_SkinThickness_0 : ", pima_data[pima_data.SkinThickness == 0].shape[0])



print(pima_data[pima_data.SkinThickness == 0].groupby('Outcome')['Age'].count())
print("Count_Insulin_0 : ", pima_data[pima_data.Insulin == 0].shape[0])



print(pima_data[pima_data.Insulin == 0].groupby('Outcome')['Age'].count())
print("Count_BMI_0 : ", pima_data[pima_data.BMI == 0].shape[0])



print(pima_data[pima_data.BMI == 0].groupby('Outcome')['Age'].count())
# removing the rows in which the “BloodPressure”, “BMI” and “Glucose” are zero.



pima_mod = pima_data[(pima_data.BloodPressure != 0) & (pima_data.BMI != 0) & (pima_data.Glucose != 0)]

print(pima_mod.shape)

pima_mod.head()
# define X and y

feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'BMI','Insulin', 'Age']



# X is a matrix, hence we use [] to access the features we want in feature_cols

X = pima_mod[feature_cols]

#X = pima_data[feature_cols]

# y is a vector, hence we use it to access 'Outcome'

y = pima_mod.Outcome

#y = pima_data.Outcome
# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# train a logistic regression model on the training set

from sklearn.linear_model import LogisticRegression



# instantiate model

logreg = LogisticRegression()



# fit model

logreg.fit(X_train, y_train)
# make class predictions for the testing set

y_pred = logreg.predict(X_test)
y_pred
# calculating accuracy ..... How many +ve & -ve cases prediction we got correctly



print('Classification Accuracy:', metrics.accuracy_score(y_test, y_pred))

print('\n')

print('The combine accuracy for finding diabetic and nondiabetic correctly  is 76.25% for model with feature_cols variables.')

print('Because it is a combine matrix & may be accuracy is more on +ve cases. So it will not provide much information to understand accuracy of + cases or - ve cases.')

print('Therefore, we are not much interested in classification accuracy for this case.')
# examine the class distribution of the testing set (using a Pandas Series method)

y_test.value_counts()
# calculating the percentage of ones

# because y_test only contains ones and zeros, we can simply calculate the mean = percentage of ones

y_test.mean()

# calculating the percentage of zeros

1 - y_test.mean()
# calculating null accuracy in a single line of code  





# only for binary classification problems coded as 0/1

max(y_test.mean(), 1 - y_test.mean())
# calculating null accuracy (for multi-class classification problems)



y_test.value_counts().head(1) / len(y_test)
#Comparing the true and predicted response values

# print the first 25 true and predicted responses

print('Test:', y_test.values[0:25])

print('Predicted:', y_pred[0:25])

print('\n')

print('The formula for probaility is sigmoid function.')

print('Whenever the probaility value of predicted model is greater than 0.5 than it will be predicted as 1.')

print('Similarly, when probability is less than 0.5. It will be predicted as 0.')





# IMPORTANT: first argument is true values, second argument is predicted values

# this produces a 2x2 numpy array (matrix)



print(metrics.confusion_matrix(y_test, y_pred))

conf=metrics.confusion_matrix(y_test, y_pred)
# save confusion matrix and slice into four pieces

confusion = metrics.confusion_matrix(y_test, y_pred)

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
# Confusion matrix visualization with seaborn 

cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)

sns.heatmap(conf,cmap = cmap,xticklabels=[' Predicted 0','Predicted 1'],yticklabels=['Actual 0','Actual 1'],annot=True, fmt="d")
# print the first 25 true and predicted responses

print('True', y_test.values[0:25])

print('Pred', y_pred[0:25])
# using float to perform true division, not integer division  (Different methods to calculate Classification Accuracy)



print((TP + TN) / float(TP + TN + FP + FN))

print('\n')

print('Classification Accuracy:', metrics.accuracy_score(y_test, y_pred))
 # Different method to calculate Classification Error

    

classification_error = (FP + FN) / float(TP + TN + FP + FN) 



print(classification_error)

print('\n')

print('Classification Error:',1 - metrics.accuracy_score(y_test, y_pred))
# Different method to calculate Sensitivity

    

sensitivity = TP / float(FN + TP)



print(sensitivity)

print('\n')

print('Sensitivity or Recall:', metrics.recall_score(y_test, y_pred))

print('\n')

print('If the sensitivity value is too low then it means. Then our model is not correct & we are not catching the diabetic +vecases.')

print('In this case, the sensitivity for model is moderate & we are able to catch or classify only 63.3% diabetic cases correctly.')

print('The remaining 26.7% are missclassified, where people having diabetic leave with out recieving proper diagnosis & worsen their situation later on.')
specificity = TN / (TN + FP)



print(specificity)
# Reverse of Specificity (1- Specificity)



false_positive_rate = FP / float(TN + FP)



print(false_positive_rate)

print('FalsePositive:',1 - specificity)

print('\n')

print('FalsePositive is opposite to Specificty, So we are not able to catch 17.35% of -ve cases (nondiabetice) correctly.')
precision = TP / float(TP + FP)



print(precision)

print('Precision:', metrics.precision_score(y_test, y_pred))
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))


#Defining a sample data to test the model



feature_cols=['Pregnancies','Glucose','BloodPressure','BMI','Insulin','Age']



data =[1,89,66,28.1,94,21]

paitentid_4=pd.DataFrame([data],columns=feature_cols)

paitentid_4.head()
predictions_diabetes=logreg.predict(paitentid_4)

print(predictions_diabetes)
# print the first 10 predicted responses

# 1D array (vector) of binary values (0, 1)

logreg.predict(X_test)[0:10]
# print the first 10 predicted probabilities of class membership

logreg.predict_proba(X_test)[0:10]
# print the first 10 predicted probabilities for class 1   ( predicting diabetic cases =1)

logreg.predict_proba(X_test)[0:10, 1]
# store the predicted probabilities for class 1

y_pred_prob = logreg.predict_proba(X_test)[:, 1]
# Plotting predicion through histogram of predicted probabilities

%matplotlib inline

import matplotlib.pyplot as plt



# 8 bins

plt.hist(y_pred_prob, bins=8)



# x-axis limit from 0 to 1

plt.xlim(0,1)

plt.title('Histogram of predicted probabilities')

plt.xlabel('Predicted probability of diabetes')

plt.ylabel('Frequency')
# predict diabetes if the predicted probability is greater than 0.3

from sklearn.preprocessing import binarize



# it will return 1 for all values above 0.3 and 0 otherwise

# results are 2D so we slice out the first column



y_pred = binarize(y_pred_prob.reshape(-1,1), 0.3) 
y_pred.shape
# probability with revised threshold =0.3



y_pred_prob[0:10]
# Outcome with revised threshold =0.3



y_pred[0:10]

# previous confusion matrix (default threshold of 0.5)

print(confusion)
 #The new confusion matrix (threshold of 0.3)

    

print(metrics.confusion_matrix(y_test, y_pred))
# sensitivity has increased (used to be 0.63)

print (52 / float(52 + 8))
 # specificity has decreased (used to be 0.82)

print(68 / float(68 + 53))
# IMPORTANT: first argument is true values, second argument is predicted probabilities



# we pass y_test and y_pred_prob

# we do not use y_pred, because it will give incorrect results without generating an error

# roc_curve returns 3 objects fpr, tpr, thresholds

# fpr: false positive rate

# tpr: true positive rate

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)



plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC curve for diabetes classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)
# IMPORTANT: first argument is true values, second argument is predicted probabilities



print(metrics.roc_auc_score(y_test, y_pred_prob))
# calculate cross-validated AUC

from sklearn.model_selection import cross_val_score

cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean()

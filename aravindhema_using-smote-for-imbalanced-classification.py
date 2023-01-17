#Imbalanced Classification Problem
#Import the necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Read the input file

dataset = pd.read_csv('../input/creditcard.csv')
dataset.head()
#Check for missing values
dataset.info()
dataset.describe()
#Find out number of values in each class. 
dataset['Class'].value_counts()
#The ratio of 0s to 1s is 1:577. This is a clear case of Imbalances class classification
X = dataset.drop('Class',axis=1)
y = dataset['Class']
X.head()
#Being imbalanced classification, we cannot ensure the predictions will be correct for the minority class
#So we will perform a SMOTE to balance out the two classes. Lets split the dataset into train and test before that
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
y_test.value_counts()
#Smote will only be applied on the training dataset

from imblearn.over_sampling import SMOTE
print('Before Oversampling: \n')
print('Count of labels with 0: {}\n'.format(sum(y_train==0)))
print('Count of labels with 1: {}\n'.format(sum(y_train==1)))
sm = SMOTE(random_state=2,k_neighbors=5)
X_train_res,y_train_res = sm.fit_sample(X_train,y_train.ravel())

print('After Oversampling \n')
print('Count of labels with 0: {}\n'.format(sum(y_train_res==0)))
print('Count of labels with 1: {}\n'.format(sum(y_train_res==1)))
#Perform a Kfold cross validation and train/test the folds using LogisticRegression algorithm
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

parameters = {'C':(0.01,0.1,2,10,50)}
lr = LogisticRegression()
clf = GridSearchCV(lr,parameters,cv=5,verbose=5)
clf.fit(X_train_res,y_train_res.ravel()) 

clf.best_params_
lr1 = LogisticRegression(C=0.1,penalty='l2',verbose=5)
lr1.fit(X_train_res,y_train_res.ravel())
from sklearn.metrics import confusion_matrix,recall_score,roc_auc_score,precision_score
y_pre = lr1.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pre)
print(cnf_matrix)
#The number of correct predictions total 84192+129 = 84321. Only 18 predictions have turned out to be false negative. 
#It is reasonable because lower the FN, (higher recall) better the accuracy of the predictions in this case
#Number of false positives (precision is low) is high
print('Recall metric in the test dataset: ',recall_score(y_test,y_pre))
print('Precision metric in the test dataset: ',precision_score(y_test,y_pre))
from sklearn.metrics import roc_curve,auc

tmp1 = lr1.fit(X_train_res,y_train_res.ravel())
y_pred_sample_score = tmp1.decision_function(X_test)

fpr,tpr,thresholds = roc_curve(y_test,y_pred_sample_score)
roc_auc = auc(fpr,tpr)

print(roc_auc)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr,tpr,'b',label = 'AOC %0.2f'% roc_auc)
plt.plot([0,1],[0,1],'r--')
plt.legend(loc='lower right')

plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
#Lets run a quick few steps to figure out what would have happened if we had not applied the SMOTE. 
#Running the train_test_Split again
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

lr1 = LogisticRegression()
lr1.fit(X_train,y_train)
y_pred_1 = lr1.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred_1)
print(cnf_matrix)
#The number of True Positives reduces from 129 to 77, The number of false negatives increased to 70 from 18
#The accuracy 85285+77 = 85362 is higher than the accuracy derived after SMOTE. So clearly accuracy does not help judge the 
#goodness of the algorithm in certain imbalanced cases as there
print('Recall metric in the test dataset: ',recall_score(y_test,y_pred_1))
#Recall reduces drastically. Reason is obvious
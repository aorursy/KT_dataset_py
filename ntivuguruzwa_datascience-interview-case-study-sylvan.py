import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



# Read the data

data = pd.read_csv('../input/predicting-churn-for-bank-customers/Churn_Modelling.csv')

data.head()
#Checking the categories in categorical variables



print(data['Geography'].unique())

print(data['Gender'].unique())
#Find blanks in data

data.info()
#create a dict file to convert string variable into numerical one

# for Gender column

gender = {'Male':0, 'Female':1}

data.Gender = [gender[item] for item in data.Gender]

data.head()
#create a dict file to convert string variable into numerical one

#For contries

geo = {'France':1, 'Spain':2, 'Germany':3}

data.Geography = [geo[item] for item in data.Geography]

data.head()
# delete the unnecessary features from dataset

data.pop('CustomerId')

data.pop('Surname')

data.pop('RowNumber')

data.head()
corr = data.corr()

sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True, annot_kws={'size':12})

heat_map=plt.gcf()

heat_map.set_size_inches(20,15)

plt.xticks(fontsize=10)

plt.yticks(fontsize=10)

plt.show()
from sklearn.model_selection import train_test_split 

train, test = train_test_split(data, test_size = 0.25)

 

train_y = train['Exited']

test_y = test['Exited']

 

train_x = train

train_x.pop('Exited')

test_x = test

test_x.pop('Exited')
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report

 

logisticRegr = LogisticRegression()

logisticRegr.fit(X=train_x, y=train_y)

 

test_y_pred = logisticRegr.predict(test_x)

confusion_matrix = confusion_matrix(test_y, test_y_pred)

print('Intercept: ' + str(logisticRegr.intercept_))

print('Regression: ' + str(logisticRegr.coef_))

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logisticRegr.score(test_x, test_y)))

print(classification_report(test_y, test_y_pred))

 

confusion_matrix_df = pd.DataFrame(confusion_matrix, ('No churn', 'Churn'), ('No churn', 'Churn'))

heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={"size": 20}, fmt="d")

heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize = 14)

heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize = 14)

plt.ylabel('True label', fontsize = 14)

plt.xlabel('Predicted label', fontsize = 14)
# Checking how many customers exited

data['Exited'].value_counts()
from sklearn.utils import resample

 

data_majority = data[data['Exited']==0]

data_minority = data[data['Exited']==1]

 

data_minority_upsampled = resample(data_minority,

replace=True,

n_samples=7963, #same number of samples as majority classe

random_state=1) #set the seed for random resampling

# Combine resampled results

data_upsampled = pd.concat([data_majority, data_minority_upsampled])

 

data_upsampled['Exited'].value_counts()
train, test = train_test_split(data_upsampled, test_size = 0.25)

 

train_y_upsampled = train['Exited']

test_y_upsampled = test['Exited']

 

train_x_upsampled = train

train_x_upsampled.pop('Exited')

test_x_upsampled = test

test_x_upsampled.pop('Exited')

 

logisticRegr_balanced = LogisticRegression()

logisticRegr_balanced.fit(X=train_x_upsampled, y=train_y_upsampled)

 

test_y_pred_balanced = logisticRegr_balanced.predict(test_x_upsampled)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logisticRegr_balanced.score(test_x_upsampled, test_y_upsampled)))

print(classification_report(test_y_upsampled, test_y_pred_balanced))

from sklearn.metrics import roc_auc_score

 

# Get class probabilities for both models

test_y_prob = logisticRegr.predict_proba(test_x)

test_y_prob_balanced = logisticRegr_balanced.predict_proba(test_x_upsampled)

 

# We only need the probabilities for the positive class

test_y_prob = [p[1] for p in test_y_prob]

test_y_prob_balanced = [p[1] for p in test_y_prob_balanced]

 

print('Unbalanced model AUROC: ' + str(roc_auc_score(test_y, test_y_prob)))

print('Balanced model AUROC: ' + str(roc_auc_score(test_y_upsampled, test_y_prob_balanced)))
from sklearn import tree

from sklearn import tree

 

# Create each decision tree (pruned and unpruned)

decisionTree_unpruned = tree.DecisionTreeClassifier()

decisionTree = tree.DecisionTreeClassifier(max_depth = 4)

 

# Fit each tree to our training data

decisionTree_unpruned = decisionTree_unpruned.fit(X=train_x, y=train_y)

decisionTree = decisionTree.fit(X=train_x, y=train_y)

 

test_y_pred_dt = decisionTree.predict(test_x)

test_y_pred_dt = decisionTree_unpruned.predict(test_x)

test_y_pred_dt = decisionTree.predict(train_x)

test_y_pred_dt = decisionTree_unpruned.predict(train_x)

print('Accuracy of unpruned decision tree classifier on train set: {:.2f}'.format(decisionTree_unpruned.score(train_x, train_y)))

print('Accuracy of unpruned decision tree classifier on test set: {:.2f}'.format(decisionTree_unpruned.score(test_x, test_y)))

print('Accuracy of decision tree classifier on train set: {:.2f}'.format(decisionTree.score(train_x, train_y)))

print('Accuracy of decision tree classifier on test set: {:.2f}'.format(decisionTree.score(test_x, test_y)))
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
clf = KNeighborsClassifier(n_neighbors = 13)

scoring = 'accuracy'

score = cross_val_score(clf, test_x, test_y, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
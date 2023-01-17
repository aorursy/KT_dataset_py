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
# Importing required libraries 

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns 

from matplotlib import gridspec
# Load the dataset from the csv file using pandas 

# best way is to mount the drive on colab and  

# copy the path for the csv file 

data = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
# Grab a peek at the data 

data.head() 
# Print the shape of the data 

# data = data.sample(frac = 0.1, random_state = 48) 

print(data.shape) 

print(data.describe()) 
# Determine number of fraud cases in dataset 

fraud = data[data['Class'] == 1] 

valid = data[data['Class'] == 0] 

outlierFraction = len(fraud)/float(len(valid)) 

print(outlierFraction) 

print('Fraud Cases: {}'.format(len(data[data['Class'] == 1]))) 

print('Valid Transactions: {}'.format(len(data[data['Class'] == 0]))) 
# Only 0.17% of all transactions are fraudulent. The data is highly imbalanced. 

# Let's apply our unbalanced models first, and if we don't get good accuracy, we can find a way to balance this dataset. 

# But first, let's implement the model without it and balance the data only when needed
print('Amount details of the fraudulent transaction') 

fraud.Amount.describe() 
# As we can clearly see from this average monetary transaction there are more fraudulent transactions. This makes this problem
# Correlation graphically gives us an idea of how features correlate with each other and can help us predict which features are most important to a forecast

# Correlation matrix 

corrmat = data.corr() 

fig = plt.figure(figsize = (12, 9)) 

sns.heatmap(corrmat, vmax = .8, square = True) 

plt.show() 
# In the heat map, we can clearly see that most of the features are not correlated with other features, but there are some features that are positively or negatively correlated with each other. For example, V2 and V5 are strongly negatively correlated with the Amount function. 

# We also see some correlation with the V20 and Amount. 

# This gives us a deeper understanding of the data available to us
# dividing the X and the Y from the dataset 

X = data.drop(['Class'], axis = 1) 

Y = data["Class"] 

print(X.shape) 

print(Y.shape) 

# getting just the values for the sake of processing  

# (its a numpy array with no columns) 

xData = X.values 

yData = Y.values 
# Using Skicit-learn to split data into training and testing sets 

from sklearn.model_selection import train_test_split 

# Split the data into training and testing sets 

xTrain, xTest, yTrain, yTest = train_test_split( 

        xData, yData, test_size = 0.2, random_state = 42) 
# I will say in advance that in the course of my testing, the Random Forest algorithm coped best with the task, so I'll start with it

# Building the Random Forest Classifier (RANDOM FOREST)  

# Random Forest

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier() 

rfc.fit(xTrain, yTrain) 

# predictions 

yPred = rfc.predict(xTest) 
# Evaluating the classifier 

# printing every score of the classifier 

# scoring in anything 

from sklearn.metrics import classification_report, accuracy_score  

from sklearn.metrics import precision_score, recall_score 

from sklearn.metrics import f1_score, matthews_corrcoef 

from sklearn.metrics import confusion_matrix 

  

n_outliers = len(fraud) 

n_errors = (yPred != yTest).sum() 

print("The model used is Random Forest classifier") 

  

acc = accuracy_score(yTest, yPred) 

print("The accuracy is {}".format(acc)) 

  

prec = precision_score(yTest, yPred) 

print("The precision is {}".format(prec)) 

  

rec = recall_score(yTest, yPred) 

print("The recall is {}".format(rec)) 

  

f1 = f1_score(yTest, yPred) 

print("The F1-Score is {}".format(f1)) 

  

MCC = matthews_corrcoef(yTest, yPred) 

print("The Matthews correlation coefficient is{}".format(MCC)) 
# printing the confusion matrix 

LABELS = ['Normal', 'Fraud'] 

conf_matrix = confusion_matrix(yTest, yPred) 

plt.figure(figsize =(12, 12)) 

sns.heatmap(conf_matrix, xticklabels = LABELS,  

            yticklabels = LABELS, annot = True, fmt ="d"); 

plt.title("Confusion matrix") 

plt.ylabel('True class') 

plt.xlabel('Predicted class') 

plt.show() 
# Let's check the rest of the algorithms

# Importing Libraries

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import classification_report

from sklearn import metrics 
# Decision Tree

decision_tree = DecisionTreeClassifier()

decision_tree.fit(xTrain, yTrain)

# predictions 

yPred = decision_tree.predict(xTest)

acc_decision_tree = round(decision_tree.score(xTrain, yTrain) * 100, 2)

acc_decision_tree
# While the decision tree

# does a good job too

# Let's take a closer look

n_outliers2 = len(fraud) 

n_errors2 = (yPred != yTest).sum() 

print("The model used is Decision Tree") 

  

acc2 = accuracy_score(yTest, yPred) 

print("The accuracy is {}".format(acc)) 

  

prec2 = precision_score(yTest, yPred) 

print("The precision is {}".format(prec)) 

  

rec2 = recall_score(yTest, yPred) 

print("The recall is {}".format(rec)) 

  

f1 = f1_score(yTest, yPred) 

print("The F1-Score is {}".format(f1)) 

  

MCC2 = matthews_corrcoef(yTest, yPred) 

print("The Matthews correlation coefficient is{}".format(MCC)) 
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(xTrain, yTrain)

y_pred = random_forest.predict(xTest)

random_forest.score(xTrain, yTrain)

acc_random_forest = round(random_forest.score(xTrain, yTrain) * 100, 2)

acc_random_forest
# As I wrote earlier, the best result is given by a Random Forest, so there is no point in describing in detail
# Gaussian Naive Bayes

gaussian = GaussianNB()

gaussian.fit(xTrain, yTrain)

y_pred = gaussian.predict(xTest)

acc_gaussian = round(gaussian.score(xTrain, yTrain) * 100, 2)

acc_gaussian

print(classification_report(yTest, y_pred))

print("accuracy:",metrics.accuracy_score(yTest, y_pred))
# Perceptron

perceptron = Perceptron()

perceptron.fit(xTrain, yTrain)

y_pred = perceptron.predict(xTest)

acc_perceptron = round(perceptron.score(xTrain, yTrain) * 100, 2)

acc_perceptron

print(classification_report(yTest, y_pred))

print("accuracy:",metrics.accuracy_score(yTest, y_pred))
# Linear SVC

linear_svc = LinearSVC()

linear_svc.fit(xTrain, yTrain)

y_pred = linear_svc.predict(xTest)

acc_linear_svc = round(linear_svc.score(xTrain, yTrain) * 100, 2)

acc_linear_svc

print(classification_report(yTest, y_pred))

print("accuracy:",metrics.accuracy_score(yTest, y_pred))
# Stochastic Gradient Descent

sgd = SGDClassifier()

sgd.fit(xTrain, yTrain)

y_pred = sgd.predict(xTest)

acc_sgd = round(sgd.score(xTrain, yTrain) * 100, 2)

acc_sgd

print(classification_report(yTest, y_pred))

print("accuracy:",metrics.accuracy_score(yTest, y_pred))
# K Nearest Neighbor:

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(xTrain, yTrain)

y_pred = knn.predict(xTest)

acc_knn = round(knn.score(xTrain, yTrain) * 100, 2)

acc_knn

print(classification_report(yTest, y_pred))

print("accuracy:",metrics.accuracy_score(yTest, y_pred))
# Support Vector Machines

svc = SVC()

svc.fit(xTrain, yTrain)

y_pred = svc.predict(xTest)

acc_svc = round(svc.score(xTrain, yTrain) * 100, 2)

acc_svc

print(classification_report(yTest, y_pred))

print("accuracy:",metrics.accuracy_score(yTest, y_pred))
# AdaBoost

from sklearn.ensemble import AdaBoostClassifier

Ada = AdaBoostClassifier(random_state=1)

Ada.fit(xTrain, yTrain)

y_pred = Ada.predict(xTest)

acc_add = round(Ada.score(xTrain, yTrain) * 100, 2)

acc_add

print(classification_report(yTest, y_pred))

print("accuracy:",metrics.accuracy_score(yTest, y_pred))
# XGBoost

import xgboost as xgb

XGB = xgb.XGBClassifier(random_state=1)

XGB.fit(xTrain, yTrain)

y_pred = XGB.predict(xTest)

acc_XGB = round(XGB.score(xTrain, yTrain) * 100, 2)

acc_XGB

print(classification_report(yTest, y_pred))

print("accuracy:",metrics.accuracy_score(yTest, y_pred))
# Now let's create a table and see what we get

import pandas as pd



Result = pd.DataFrame({

    'MODEL': ['Support Vector Machines', 'KNN', 'Logistic Regression',

              'Random Forest','Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent','Decision Tree', 'AdaBoost', 'XGBoost'],

    'SCORE': [acc_linear_svc, acc_knn, acc_sgd,acc_random_forest, 

              acc_gaussian, acc_perceptron, 

              acc_sgd, acc_decision_tree, acc_XGB, acc_add]})

Result = Result.sort_values(by='SCORE', ascending=False)

Result = Result.set_index('SCORE')

Result.head(10)
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Remove the duplicate record. 
train = train.drop_duplicates()
test = test.drop_duplicates()

#Change Categorical variable Male - 0 and Female - 1
train['Sex'] = train['Sex'] .map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'] .map({'male': 0, 'female': 1})
train.head(3)
train_sub = pd.DataFrame(train[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Survived']])
train_sub.head(3)
test_sub = pd.DataFrame(test[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare']])
test_sub.head(3)
#Data Cleaning Process
# check if any columns in data set is blank
print("Does training data have null values: ",train_sub.isnull().any().any()) 
print("Does test data have null values: ",test_sub.isnull().any().any())
# To check which column of training set has NaN
print("Column with NA values: ",train_sub.columns[train_sub.isna().any()].tolist())

#To check number of non null values
train_sub.info()
# Replace NaN values with mean for training data
train_sub["Age"].fillna(train_sub["Age"].median(), inplace=True)


# To check which columns of test set has NaN
print("Column with NA values: ",test_sub.columns[test_sub.isna().any()].tolist())

#To check number of non null values
test_sub.info()
# Replace NaN values with mean for training data
test_sub["Age"].fillna(test_sub["Age"].median(), inplace=True)
test_sub['Fare'].fillna(test_sub["Fare"].median(), inplace=True)
#test['Cabin'].fillna(0, inplace=True)
import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))
corr = train_sub.corr()
sns.heatmap(corr, ax=ax, annot=True, fmt=".2f")
train_sub.head(3)
test_sub.head(3)
test_sub_x = test_sub.iloc[:, [1,2,3,4,5,6]].values
x = train_sub.iloc[:, [1,2,3,4,5,6]].values
y = train_sub.iloc[:, -1].values
#Split train data into training set and test set
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.25,random_state=0)
print("Train set: ", train_x.shape, train_y.shape)
print("Test set: ", test_x.shape, test_y.shape)
#Classifier to implement KNN
from sklearn.neighbors import KNeighborsClassifier
#initialize k  to 4
k = 6
# Fitting the training model
regressor = KNeighborsClassifier(n_neighbors = k).fit(train_x, train_y)
regressor
#Predict based on test data
y_hat = regressor.predict(test_x)
y_hat[0:3]
print("Test set: ", test_y.shape)
print("Y Pred: ", y_hat.shape)
#Accuracy Evaluation
from sklearn import metrics
print("Train set accuracy: ",metrics.accuracy_score(train_y, regressor.predict(train_x)))
print("Test set accuracy: ",metrics.accuracy_score(test_y, y_hat))
# Calculate accuracy of KNN for different values of k
ks = 10
#initialize mean_acc and std_acc
mean_acc = np.zeros(ks-1)
std_acc = np.zeros(ks-1)
ConfusionMatrix = []

for n in range (1,ks):
    regressor = KNeighborsClassifier(n_neighbors = n).fit(train_x, train_y)
    yhat = regressor.predict(test_x)
    mean_acc[n-1] = metrics.accuracy_score(test_y,yhat)
    std_acc[n-1]=np.std(yhat==test_y)/np.sqrt(yhat.shape[0])
    
mean_acc
#Making Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_y, y_hat)
print("cm :",cm)
#124+42= 166 correct predicted value and 15+42=57 wrong predicted value

x = train_sub.iloc[:, [1,2,3,4,5,6]]
y = train_sub.iloc[:, [-1]]

#Data standadization - converted to zscore (normalizing the data)
from sklearn import preprocessing
x = preprocessing.StandardScaler().fit(x).transform(x.astype('float'))
x[0:3]

#Split train data into training set and test set
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.25,random_state=0)


#Fitting Logistic Regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(train_x, train_y.values.reshape(-1,))


# Score function of sklearn can quickly assess the model performance
classifier.score(train_x, train_y)
#Around 80% of the observation are correctly classified!

#Predicting the test set
y_pred = classifier.predict(test_x)
#Accuracy Evaluation
from sklearn import metrics
print("Train set accuracy: ",metrics.accuracy_score(train_y, classifier.predict(train_x)))
print("Test set accuracy: ",metrics.accuracy_score(test_y, y_pred))

print("Test set Precision:",metrics.precision_score(test_y, y_pred))
print("Test set Recall:",metrics.recall_score(test_y, y_pred))

#Making Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_y, y_pred)
print("cm :",cm)
#118+59= 177 correct predicted value and 21+25=46 wrong predicted value
#ROC Curve
y_pred_proba = classifier.predict_proba(test_x)[::,1]
fpr, tpr, _ = metrics.roc_curve(test_y,  y_pred_proba)
auc = metrics.roc_auc_score(test_y, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
#F Score of Logistic Model
from sklearn.metrics import f1_score
print('F1 score:', f1_score(test_y, y_pred))
x = train_sub.iloc[:, [1,2,3,4,5,6]].values
y = train_sub.iloc[:, -1].values
#Split train data into training set and test set
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.25,random_state=0)
#Classifier to implement RandomForest
from sklearn.ensemble import RandomForestClassifier

regressor = RandomForestClassifier(n_estimators = 1500, random_state = 0)
regressor.fit(train_x, train_y)
#  Predict a new result , 165K expected
y_pred = regressor.predict(test_x)
y_pred[0:3]
#Accuracy Evaluation
from sklearn import metrics
print("Train set accuracy: ",metrics.accuracy_score(train_y, regressor.predict(train_x)))
print("Test set accuracy: ",metrics.accuracy_score(test_y, y_pred))

print("Test set Precision:",metrics.precision_score(test_y, y_pred))
print("Test set Recall:",metrics.recall_score(test_y, y_pred))
#Making Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_y, y_pred)
print("cm :",cm)
#118+59= 177 correct predicted value and 21+25=46 wrong predicted value for Logistic model
#128+53=181 correct values and 11+31=42 wrong predicted value for KNN model
yhat = regressor.predict(test_sub_x)
y_pred[0:3]
#Save output to a dataframe
output = pd.DataFrame(test_sub['PassengerId'])
output['Survived'] = pd.DataFrame(yhat, columns = ["Survived"])
output.head(3)
#Save DataFrame to csv file
output.to_csv('TitanicSurvival.csv', encoding='utf-8', index=False)

# Load libraries

import pandas as pd

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

# load dataset

pima = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")

pima.head()
feature_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age','Glucose','BloodPressure','DiabetesPedigreeFunction']

X = pima[feature_cols] # Features

y = pima.Outcome # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# Create Decision Tree classifer object

clf = DecisionTreeClassifier()



# Train Decision Tree Classifer

clf = clf.fit(X_train,y_train)



#Predict the response for test dataset

y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100) #Evaluation
from sklearn.naive_bayes import GaussianNB

from sklearn.datasets import load_iris



irisGNB = GaussianNB()   #define classifier



iris = load_iris()   #load data set

irisGNB.fit(iris.data, iris.target)   #fit data set to the tree (training)



pred_class = irisGNB.predict([[6.3,2.9,5.6,1.8]]) #predict the class of any data



#bonus

y_predict = irisGNB.predict(iris.data)

l1 = [(x,y,z) for x,y,z in zip(iris.data,iris.target,y_predict) if y!=z]

print(l1)
#Import scikit-learn dataset library

from sklearn import datasets



#Load dataset

wine = datasets.load_wine()
# print the names of the 13 features

print("Features: ", wine.feature_names)



# print the label type of wine(class_0, class_1, class_2)

print("Labels: ", wine.target_names)
print(wine.target)
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3,random_state=109) # 70% training and 30% test
#Import Gaussian Naive Bayes model

from sklearn.naive_bayes import GaussianNB



#Create a Gaussian Classifier

gnb = GaussianNB()



#Train the model using the training sets

gnb.fit(X_train, y_train)



#Predict the response for test dataset

y_pred = gnb.predict(X_test)
print('Accuracy',metrics.accuracy_score(y_test,y_pred))
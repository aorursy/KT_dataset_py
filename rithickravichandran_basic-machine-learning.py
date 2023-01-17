# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# importing all the necessary packages 

from sklearn.datasets import load_iris #Iris dataset is inbuilt in scikit-learn. Doesn't require to import dataset

from sklearn import metrics #For checking the model accuracy

from sklearn.model_selection import train_test_split #For Splitting the training set and test set

from sklearn.neighbors import KNeighborsClassifier #To use the K-Nearest Neighbors Classifier Algorithm

from sklearn.linear_model import LogisticRegression #To use the Logistic Regression Algorithm

from sklearn.tree import DecisionTreeClassifier#To use the Decision Tree Classifier Algorithm

from sklearn import svm #To use the Support Vector Machine Alogorithm



iris = load_iris() #Load the dataset

x=iris.data #Storing the data features in X

y=iris.target #Storing the output of training data in y
feature_name=iris.feature_names

print(feature_name)

# These are the Features present in the dataset
target_name=iris.target_names

print(target_name)

# These are the types of flowers in the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print(X_train.shape)

print(X_test.shape)
X_train
X_test
irisknnmodel = KNeighborsClassifier(n_neighbors=3) #create object for the model

irisknnmodel.fit(X_train,y_train) #Train the model with Training Data
y_prediction=irisknnmodel.predict(X_test) #Now predict the result for test data

print("The Accuracy of K-Nearest nerighbours Classifier is",metrics.accuracy_score(y_test, y_prediction)) 

#Check the accuracy with the test output and predicted output
irislrmodel = LogisticRegression(random_state=0)#create object for the model

irislrmodel.fit(X_train,y_train)#Train the model with Training Data
y_prediction=irislrmodel.predict(X_test) #Now predict the result for test data

print("The Accuracy of Logistic Regression is",metrics.accuracy_score(y_test, y_prediction))

#Check the accuracy with the test output and predicted output
irisdscmodel=DecisionTreeClassifier()#create object for the model

irisdscmodel.fit(X_train,y_train)#Train the model with Training Data
y_prediction=irisdscmodel.predict(X_test)#Now predict the result for test data

print("The Accuracy of Decision Tree Classifier is",metrics.accuracy_score(y_test, y_prediction))

#Check the accuracy with the test output and predicted output
irissvmmodel=svm.SVC()#create object for the model

irissvmmodel.fit(X_train,y_train)#Train the model with Training Data
y_prediction=irissvmmodel.predict(X_test)#Now predict the result for test data

print("The Accuracy of Support Vector Machine (SVM) is",metrics.accuracy_score(y_test, y_prediction))

#Check the accuracy with the test output and predicted output
sample=[[3,2,2,3], [7,3.2,4.7,1.4]]

y_prediction=irissvmmodel.predict(sample)

print(y_prediction)
prediction_targets=[iris.target_names[p] for p in y_prediction]

print("Predictions:",prediction_targets)
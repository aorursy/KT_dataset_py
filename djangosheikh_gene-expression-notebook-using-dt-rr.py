import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.ensemble import RandomForestClassifier

#loading data
Train_Data = pd.read_csv("../input/gene-expression/data_set_ALL_AML_train.csv")
Test_Data = pd.read_csv("../input/gene-expression/data_set_ALL_AML_independent.csv")
Actual = pd.read_csv("../input/gene-expression/actual.csv")
#checking for null values
print(Train_Data.isna().sum().max())
print(Test_Data.isna().sum().max())
Train_Data.head()
Test_Data.head()
#checking for object features in the dataset
data_object=Train_Data.select_dtypes(include=['object']) 
data_object.head()
#checking for non-object features
data_float_train = Train_Data.select_dtypes(include=['int64','int32','float64'])
data_float_test=Test_Data.select_dtypes(include=['int64','int32','float64'])
data_float_test.head()
#concat all the numerical features
column_names_test = list(data_float_test.columns.values)
column_names_train = list(data_float_train.columns.values)
patients=column_names_train+column_names_test
data=pd.concat([data_float_train,data_float_test],axis=1)[patients]
data.head()
# Transpose so that each row matches a patient
data=data.T
data.head()
data["patient"] = pd.to_numeric(patients)
data.head()
# AML is 1, MML is 0
Actual["cancer"]= pd.get_dummies(Actual.cancer, drop_first=True)
# add the cancer column to train data
Data = pd.merge(data, Actual, on="patient")
# split in train and test, firts 38 are train, the rest is test
Train_Data, Test_Data = Data.iloc[:39,:], Data.iloc[39:,:]
Train_Data.head()
X_train, y_train= Train_Data.drop(columns=["cancer"]), Train_Data["cancer"]
X_test, y_test = Test_Data.drop(columns=["cancer"]), Test_Data["cancer"]
#using Decision Tree classifier
# Function to perform training with giniIndex. 
def train_using_gini(X_train, X_test, y_train): 

	# Creating the classifier object 
	clf_gini = DecisionTreeClassifier(criterion = "gini", 
			random_state = 100,max_depth=3, min_samples_leaf=5) 

	# Performing training 
	clf_gini.fit(X_train, y_train) 
	return clf_gini 
	
# Function to perform training with entropy. 
def tarin_using_entropy(X_train, X_test, y_train): 

	# Decision tree with entropy 
	clf_entropy = DecisionTreeClassifier( 
			criterion = "entropy", random_state = 100, 
			max_depth = 3, min_samples_leaf = 5) 

	# Performing training 
	clf_entropy.fit(X_train, y_train) 
	return clf_entropy 


# Function to make predictions 
def prediction(X_test, clf_object): 

	# Predicton on test with giniIndex 
	y_pred = clf_object.predict(X_test) 
	print("Predicted values:") 
	print(y_pred) 
	return y_pred 
	
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
	
	print("Confusion Matrix: ", 
		confusion_matrix(y_test, y_pred)) 
	
	print ("Accuracy : ", 
	accuracy_score(y_test,y_pred)*100) 
	
	print("Report : ", 
	classification_report(y_test, y_pred)) 

# Driver code 
def main(): 
	
	# Building Phase 
	X_train, y_train,X_test, y_test= Train_Data.drop(columns=["cancer"]), Train_Data["cancer"],Test_Data.drop(columns=["cancer"]), Test_Data["cancer"]
	clf_gini = train_using_gini(X_train, X_test, y_train) 
	clf_entropy = tarin_using_entropy(X_train, X_test, y_train) 
	
	# Operational Phase 
	print("Results Using Gini Index:") 
	
	# Prediction using gini 
	y_pred_gini = prediction(X_test, clf_gini) 
	cal_accuracy(y_test, y_pred_gini) 
	
	print("Results Using Entropy:") 
	# Prediction using entropy 
	y_pred_entropy = prediction(X_test, clf_entropy) 
	cal_accuracy(y_test, y_pred_entropy) 
	
	
# Calling main function 
if __name__=="__main__": 
	main() 

#using Random forest
X_train, y_train,X_test, y_test= Train_Data.drop(columns=["cancer"]), Train_Data["cancer"],Test_Data.drop(columns=["cancer"]), Test_Data["cancer"]
# creating a RF classifier 
clf = RandomForestClassifier(n_estimators = 100)   
  
# Training the model on the training dataset 
# fit function is used to train the model using the training sets as parameters 
clf.fit(X_train, y_train) 
  
# performing predictions on the test dataset 
y_pred = clf.predict(X_test) 
  
# metrics are used to find accuracy or error 
from sklearn import metrics   
print() 
  
# using metrics module for accuracy calculation 
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

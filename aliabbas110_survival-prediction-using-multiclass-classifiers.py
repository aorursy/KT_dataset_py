import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import Normalizer

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

# from sklearn import cross_val_score

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
# Preprocessing by AZEEM & NEHA 

#Creating variable for labelEncoder

le = LabelEncoder()

#Calling & Assiging Dataset

tdb = pd.read_csv("../input/train.csv",nrows = 500)

#Selecting Desired Colomns from Dataset

X= tdb[['Sex', 'Age', 'Fare']]

#Converting Gender colomn to Binary

X.iloc[:, 0] = le.fit_transform(X.iloc[:,0])

#Assigning Play colomn to variable Y

Y = tdb['Survived']

#Assigning desired colomns to variable x for Filling missing values with mean

x = tdb[['Age', 'Fare']]

imp=Imputer(missing_values=np.nan,strategy="mean")

X[['Age', 'Fare']]=imp.fit_transform(x)

#Scaling values to range between (0 - 1)

scaler = MinMaxScaler(feature_range = (0, 1))

X[['Age', 'Fare']] = scaler.fit_transform(X[['Age', 'Fare']])

#TRAIN TEST SPLIT DATA

X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size = 0.2, random_state = 0)

print("Training data(feature set): ",X_TRAIN.shape,"\nTraining data (label set): ",Y_TRAIN.shape)

print("Testing data(feature set): ",X_TEST.shape,"\nTesting data (label set): ",Y_TEST.shape)

print("Feature set\n",X)

print("Label set\n",Y)
#Data classification and model training by ALI AND MEHROSH



#generalized function to get any model and train on it and return the score

def get_score(model,X_train,X_test,Y_train,Y_test):

    model.fit(X_train,Y_train)

    return model.score(X_test,Y_test)

    



#K-folds cross validation techinique implementation

#This technique is used in order to see whats the maximum score of any model on different K-Folds

#data is divided into K number of folds and every time one fold is used for testing and the model gets trained on rest of the folds

#returns a list of scores of length K, the highest mean of list for any model depicts that it has the highest accuracy on classifying the data

def cross_validation(model,X,y,folds=3):

    kf=KFold(n_splits=folds)

    scores=[]

    for train_index,test_index in kf.split(X):

        X_train,X_test=X.iloc[train_index],X.iloc[test_index]

        Y_train,Y_test=y.iloc[train_index],y.iloc[test_index]

        scores.append(get_score(model,X_train,X_test,Y_train,Y_test))

    return scores

    

#Creating a list of models

models=[('K Nearest Neighbors',KNeighborsClassifier(n_neighbors=10)),('SVM',SVC()),('Naive bayes',GaussianNB()),('Decision Tree',DecisionTreeClassifier())]



#applying the above function to every model and plotting the scores of every model for performance comparison 

for name,i in models:

    scores=cross_validation(i,X,Y)

    print(name,scores)

    plt.plot(scores,label=name)

    plt.legend()



#Function to generate Confusion matrix and accuracy score for every model based on the train_test_split

#technique, with test_size of 30% by default

def model_analysis(model,X,y,n=0.3):

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=n)

    model.fit(X_train,y_train)

    predicted=model.predict(X_test)

    predicted=pd.DataFrame(predicted)

    predicted.to_csv('prediction.csv',index=False)

    print(confusion_matrix(y_test,predicted))

    print("Accuracy Score :",accuracy_score(y_test,predicted)*100,"%")

    
for name,i in models:

    print(name)

    model_analysis(i,X,Y,0.2)

    print("----------------")

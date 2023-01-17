# import LIBRARIES

import pandas as pd

import numpy as np



# importing lib and Models

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

# dataset spliting

from sklearn.cross_validation import train_test_split

from sklearn.cross_validation import cross_val_score

# visualization

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns





# metrics

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



data = pd.read_csv("../input/sample-insurance-claim-prediction-dataset/insurance2.csv")

data.head(6)
#test train split

x = data.iloc[:,:-1]

y = data.iloc[:,7]

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
# Models for Heart Disease Dataset classification

models = []



def classification_Models(X_train,X_test, y_train, y_test ):

    models.append( ('LR',  LogisticRegression()) )

    models.append( ('DTC',DecisionTreeClassifier()) )

    models.append( ('KNN', KNeighborsClassifier()) )

    models.append( ('NB',  GaussianNB()) )

    models.append( ('SVM',  SVC()) )

    modelresults = []

    modelnames = []

    for name,model in models:

        v_results = cross_val_score(model, X_train, y_train, cv = 3, 

                                     scoring='accuracy', n_jobs = -1, verbose = 0)

        print(name,v_results.mean())

        modelresults.append(v_results)

        modelnames.append(name)

        

    print(modelresults)

    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.set_xticklabels(modelnames)

    plt.boxplot(modelresults)

        

classification_Models(X_train,X_test, y_train, y_test)
# Evaluating and predicting models





for name,model in models:

    trainedmodel = model.fit(X_train,y_train)

    

    # prediction

    y_predict = trainedmodel.predict(X_test)

    

    accuracy = accuracy_score(y_test,y_predict)

    classreport = classification_report(y_test,y_predict)

    confusnMatrix = confusion_matrix(y_test,y_predict)

    

    print('The accuracy: {}'.format(accuracy))

    print('The Classification Report:\n {}'.format(classreport))

    print('The Confusion Matrix:\n {}'.format(confusnMatrix))

    
#importing basic libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
#Importing File and seeing top 5rows

Gdf=pd.read_csv('../input/heart-disease-uci/heart.csv')

Gdf.head()
#Getting information about the data

Gdf.info()
#Statistical summary of data

Gdf.describe()
#List of columns

Gdf.columns
#Bar Graph for disease comparison based on Gender

pd.crosstab(Gdf.sex,Gdf.target).plot(kind="bar",figsize=(10,5),color=['blue','red' ])

plt.xlabel('Sex (0 = Female, 1 = Male)')

plt.xticks(rotation=0)

plt.legend(["Not Have Disease", "Have Disease"])

plt.ylabel('Frequency')

plt.show()
#Obtaining the values for input and output

y=Gdf.sex.values



#dropping everything except 'sex' column

x_Gdf=Gdf.drop(["sex"], axis=1)



#Normalizing the data to obtain the input x through MinMax Scaling

x=(x_Gdf-np.min(x_Gdf)/np.max(x_Gdf)-np.min(x_Gdf)).values
#Importing the library for train_test_split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)



#Printing the number of training and testing data sets

print('train data set:{}'.format(x_train.shape))

print('test data set:{}'.format(x_test.shape))
#Machine Learning Models

#Logistic Regression

from sklearn.linear_model import LogisticRegression

LR=LogisticRegression()

LR.fit(x_train,y_train)

Accuracy_LR=LR.score(x_test,y_test)

print('Accuracy Score',Accuracy_LR)
#K-Nearest Neighbor Classifier

from sklearn.neighbors import KNeighborsClassifier

score_list=[]

best_score=0

best_k_value=0

for each in range(1,15):

    KNN=KNeighborsClassifier(n_neighbors=each)

    KNN.fit(x_train,y_train)

    score_list.append(KNN.score(x_test,y_test))

    if KNN.score(x_test,y_test)>best_score:

        Accuracy_KNN=KNN.score(x_test,y_test)

        best_k_value=each

plt.plot(score_list)

plt.xlabel('K-values')

plt.ylabel('Accuracy')

plt.show

print("Accuracy Score ", Accuracy_KNN)

print("Best accuracy's k value is ", best_k_value)
#Naive Bayes Classificatiion

from sklearn.naive_bayes import GaussianNB

GNB=GaussianNB()

GNB.fit(x_train,y_train)

Accuracy_GNB=GNB.score(x_test,y_test)

print('Accuracy score of Naive Bayes',Accuracy_GNB)
#Support Vector Machine

from sklearn.svm import SVC

SVM=SVC()

SVM.fit(x_train,y_train)

Accuracy_SVM=SVM.score(x_test,y_test)

yp=SVM.predict(x_test)

print('Accuracy score for SVM',Accuracy_SVM)
#Decision Tree Classification

from sklearn.tree import DecisionTreeClassifier

Dtree=DecisionTreeClassifier(random_state=50)

Dtree.fit(x_train,y_train)

Accuracy_Dtree=Dtree.score(x_test,y_test)

print('Accuracy score of Decision Tree',Accuracy_Dtree)
#Random Forest Classification

from sklearn.ensemble import RandomForestClassifier

RFC=RandomForestClassifier()

RFC.fit(x_train,y_train)

Accuracy_RFC=RFC.score(x_test,y_test)

print('Accuracy of Random Forest Classifier',(Accuracy_RFC))
#Visualizing the overall accuracies of performed models through graph

Supervised_learning_Models=["Log Reg", "KNN", "Naive Bayes","SVM", "Decision Tree", "Random Forest"]

Overall_Accuracy_Score=[Accuracy_LR,Accuracy_KNN,Accuracy_GNB,Accuracy_SVM,Accuracy_Dtree,Accuracy_RFC]

sns.barplot(x=Supervised_learning_Models,y=Overall_Accuracy_Score)

plt.xlabel("Supervised Models")

plt.ylabel("Accuracy")

plt.title("Overall Accuracy Score Graph")

plt.show()
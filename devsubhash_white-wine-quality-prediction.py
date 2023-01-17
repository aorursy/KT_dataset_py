# getting started with the model 

# importing required libraries/packages 



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



#import time for training details

from time import time

t0 = time()



import warnings

warnings.filterwarnings('ignore')
# Importing and Reading the Dataset

df_wine= pd.read_csv('../input/winequality-white.csv')
# check to see if there are any missing entries

df_wine.info()
df_wine.head().iloc[:5]
#getting column names

df_wine.columns
#checking Datatypes

df_wine.dtypes
#correlation map for features

f,ax = plt.subplots(figsize=(9, 9))

ax.set_title('Correlation map for variables')

sns.heatmap(df_wine.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax,cmap="PuRd")
#Getting an idea about the distribution of wine quality 

p = sns.countplot(data=df_wine, x = 'quality', palette='muted')
df_wine['quality'].describe()
#'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',

#'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',

# 'pH', 'sulphates', 'alcohol', 'quality'

#Getting an idea about the distribution of wine quality 



p = sns.barplot(data=df_wine, x = 'quality',y='alcohol', palette='muted')
p = sns.barplot(data=df_wine, x = 'quality',y='volatile acidity',palette='muted')
p = sns.barplot(data=df_wine, x = 'quality',y='fixed acidity',palette='muted')
#Grouping the wine based on grade

# Defining 'grade' of wine



#Good wine

df_wine['grade'] = 1 



#Bad wine

df_wine.grade[df_wine.quality < 6.5] = 0 



#sns.set(style="whitegrid")

#p = sns.countplot(data=df_wine, x='grade', palette='muted')



#set plotsize and colors



plt.figure(figsize = (6,6))

colors = ['lightcoral', 'rosybrown']



labels = df_wine.grade.value_counts().index

plt.pie(df_wine.grade.value_counts(), autopct='%1.1f%%',colors=colors)

plt.legend(labels, loc="Best")

plt.axis('equal')

plt.title('White Wine Quality Distribution')

plt.show()

#Show mean quality of white wine and quality distribution



print('The amount of good quality white wine is ',round(df_wine.grade.value_counts(normalize=True)[1]*100,1),'%.')

print("mean white wine quality = ",df_wine["quality"].mean())
# plot to see how pH is varying in the grade of white wine



plt.figure(figsize=(6,6))

ax = sns.lineplot(x="pH", y="quality", hue="grade", data=df_wine,markers=True)
df_wine['grade'].value_counts() #prints counts of good and bad white wine
#Checking once more for column names

df_wine.columns
#Defining X and y

X = df_wine.drop(['quality'], axis=1)

y = df_wine['quality']
# creating dataset split for prediction

from sklearn.model_selection import train_test_split

X_train, X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42) # 80-20 split



# Checking split 

print('X_train:', X_train.shape)

print('y_train:', y_train.shape)

print('X_test:', X_test.shape)

print('y_test:', y_test.shape)
# 1. Using Random Forest Classifier

t0 = time()

# Load random forest classifier 

from sklearn.ensemble import RandomForestClassifier



# Create a random forest Classifier

clf = RandomForestClassifier(n_jobs=2, random_state=0)



# Train the Classifier/fitting the model

clf.fit(X_train, y_train)



# predict the response

y_pred = clf.predict(X_test)

acc_rf = round(clf.score(X_test,y_test) * 100, 2)

#Import scikit-learn metrics module for accuracy calculation

from sklearn import metrics



# evaluate accuracy

print("Random Forest Classifier Accuracy:",metrics.accuracy_score(y_test, y_pred)*100,"%")

print('Training time', round(time() - t0, 3), 's')
#2. Gaussian Naive Bayes Classifier

t0 = time()

#Import Gaussian Naive Bayes model

from sklearn.naive_bayes import GaussianNB



#Create a Gaussian Classifier

gnb = GaussianNB()



# Train the Classifier/fitting the model

gnb.fit(X_train, y_train)



# predict the response

y_pred = gnb.predict(X_test)

acc_gnb = round(gnb.score(X_test,y_test) * 100, 2)



#Import scikit-learn metrics module for accuracy calculation

from sklearn import metrics



# evaluate accuracy

print("Naive Bayes Accuracy:",metrics.accuracy_score(y_test, y_pred)*100,"%")

print('Training time', round(time() - t0, 3), 's')
#import Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

from sklearn import tree



# Create Decision Tree classifer object

clf = DecisionTreeClassifier(max_depth=10)



# Train the Classifier/fitting the model

clf = clf.fit(X_train,y_train)



# predict the response

y_pred = clf.predict(X_test)

acc_dt = round(clf.score(X_test,y_test) * 100, 2)

#Import scikit-learn metrics module for accuracy calculation

from sklearn.metrics import accuracy_score 



# evaluate accuracy

print ("Decision Tree Accuracy:", metrics.accuracy_score(y_test, y_pred)*100,"%")

print('Training time', round(time() - t0, 3), 's')
#kNN

import sys, os



# Import kNN classifier

from sklearn.neighbors import KNeighborsClassifier



# instantiate learning model (k = 3)

knn = KNeighborsClassifier(n_neighbors=3)



# Train the Classifier/fitting the model

knn.fit(X_train, y_train)



# predict the response

y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_test,y_test) * 100, 2)

#Import scikit-learn metrics module for accuracy calculation

from sklearn import metrics



# evaluate accuracy

print("kNN Accuracy:",metrics.accuracy_score(y_test, y_pred)*100,"%")

print('Training time', round(time() - t0, 3), 's')
#Support Vector Machines trial

import sys, os



#Import svm model

from sklearn import svm

from sklearn.svm import SVC



#Create a svm Classifier

clf = SVC(C=1, kernel='rbf')



# Train the Classifier/fitting the model

clf.fit(X_train, y_train)



# predict the response

y_pred = clf.predict(X_test)

acc_svm = round(clf.score(X_test,y_test) * 100, 2)



# evaluate accuracy

print("SVM Accuracy:",metrics.accuracy_score(y_test, y_pred)*100,"%")

print('Training time', round(time() - t0, 3), 's')
# visualizing accuracies for all ML Algorithms using Matplotlib

predictors_group = ('Random Forest', 'GaussianNB', 'DecisionTree','kNN','SVM')

x_pos = np.arange(len(predictors_group))

accuracies1 = [acc_rf, acc_gnb, acc_dt,acc_svm,acc_knn]

    

plt.bar(x_pos, accuracies1, align='center', alpha=0.5, color='purple')

plt.xticks(x_pos, predictors_group, rotation='vertical')

plt.ylabel('Accuracy (%)')

plt.title('Classifier Accuracies')

plt.show()
#printing top three accuracies



print('Decision Tree:', acc_dt,'%')

print('Random Forest:', acc_rf,'%')

print('GaussianNB:',acc_gnb,'%')
# importing the model for prediction



from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB



# creating list of tuple wth model and its name  

models = []

models.append(('DT',DecisionTreeClassifier()))

models.append(('RF',RandomForestClassifier()))

models.append(('GNB',GaussianNB()))
# Import Cross Validation 

from sklearn.model_selection import cross_val_score



# simulate splitting a dataset of 1000 observations into 5 folds

from sklearn.model_selection import KFold

kf = KFold(n_splits=5, random_state=42, shuffle=True)

kf.get_n_splits(X)

# print(kf)



acc = []   # All Algorithm/model accuracies

names = []    # All model name



for name, model in models:

    

    acc_of_model = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy') # kFolds =5 without shuffling

    

    acc.append(acc_of_model) # appending Accuray of different model to acc List

    

    names.append(name)# appending name of models

    Acc =name,round(acc_of_model.mean()*100,2) # printing Output 

    print(Acc)
# Plotting all accuracies together for comparison



labels = ['Decision Tree', 'Random Forest','Gaussian NB']



NoCV =[69.49 ,77.65,66.43] # accuracy before Cross Validation

CV=[69.24, 75.14, 65.8] # accuracy after Cross Validation



x = np.arange(len(labels))  # the label locations

width = 0.25  # the width of the bars



f, ax = plt.subplots(figsize=(8,6)) 

p1 = ax.bar(x - width/2, CV, width, label='After Cross Validation', color='purple')

p2 = ax.bar(x + width/2, NoCV, width, label='Before Cross Validation', color='m')



# Add some text for labels and title 

ax.set_ylabel('Accuracies')

ax.set_title('Accuracy comparison')

ax.set_xticks(x)

plt.xticks()

ax.set_xticklabels(labels)

ax.legend(loc='top right')

plt.show()
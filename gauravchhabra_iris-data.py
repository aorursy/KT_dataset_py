# Importing libraries - Data wrangling

import numpy as np

import pandas as pd
# Importing libraries - Data Vusualizaton

import seaborn as sns

import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix
import os

print(os.listdir("../input/"))
# Loading IRIS Data Set

rawdata = pd.read_excel("../input/iris.xlsx")
# Understanding/Exploring Data

rawdata.head(10)
# Checking number of columns and rows in data

rawdata.shape
# Checking data type

rawdata.dtypes
# More inforamation about data

rawdata.describe()
# Checking null values in data

rawdata.isnull().sum()
# Checking data count species wise

rawdata.groupby('species').count()
# Checking data count species wise

rawdata.groupby('species').size()
# Some more detail about data

rawdata.info()
## Visualization



# Box & Whisker Plot

rawdata.plot.box(figsize=(8,6), title = 'Box & Whisker Plot - IRIS Data', legend = True);

plt.xlabel("Species");

plt.ylabel("CM");
# Petal Length & Width and Sepal Length and Width

plt.figure(figsize=(15,10));

plt.subplot(2,2,1);

sns.violinplot(x='species',y='petallength',data=rawdata);

plt.subplot(2,2,2);

sns.violinplot(x='species',y='petalwidth',data=rawdata);

plt.subplot(2,2,3);

sns.violinplot(x='species',y='sepallength',data=rawdata);

plt.subplot(2,2,4);

sns.violinplot(x='species',y='sepalwidth',data=rawdata);
# Correlation among columns

#    Accoridng to correlation matrix Petal lenth & Petal width are highly correlated and have positive relation.

corr_matrix = rawdata.corr()

plt.figure(figsize=(8,8))

sns.heatmap(corr_matrix, annot = True);
# Pair plot

sns.pairplot(rawdata, hue="species");
## Preparing Data for models



# Creating X & y variables

X = rawdata.drop('species', axis = 'columns')

y = rawdata['species']
# Checking variables data

print("Sepal & Pedal Data (CM)")

print(X.head())

print("Species Data")

print(y.head())
# Importing library - for splitting data into Training and Testing

from sklearn.model_selection import train_test_split
# Splitting Data - 80% Training data & 20% Testing Data

X_train, X_test, y_train, y_test = train_test_split(X, y,

                                                    test_size = 0.20,

                                                    random_state = 0

                                                    )
# No of row and columns after slipliting data for training and testing 

print("X_Train Shape is", X_train.shape, "& X_Test Shape is", X_test.shape )

print("Y_Train Shape is", y_train.shape, "& Y_test Shape is", y_test.shape)
# Importing logistic regression from Sklearn's linear model liberary

from sklearn.linear_model import LogisticRegression
# Implementing logistic regression

l_reg = LogisticRegression()   

l_reg.fit(X_train, y_train)
# Predecting "y" from X test on the basis of the X_train and y_train

y_pred_lr = l_reg.predict(X_test)
# Checking Accuracy Score

from sklearn.metrics import accuracy_score

accuracy_score_LOG = accuracy_score(y_pred_lr,y_test)

print("Accuracy Score for Logistic Regression Model is", accuracy_score(y_pred_lr,y_test))
# Checking F1 Score

from sklearn.metrics import f1_score

F1_Score_LOG = f1_score(y_test,y_pred_lr, average='macro')

print("F1 Score for Logistic Regression Model is", f1_score(y_test,y_pred_lr, average='macro'))
# Confusion matrix

from sklearn.metrics import confusion_matrix

confusin_matrix_LOG = confusion_matrix(y_test, y_pred_lr)

plt.figure(7);

plt.title('Confusion Matrix - Logistic Regression')

sns.heatmap(confusin_matrix_LOG,annot = True);
# Importing GaussianNB from Sklearn's Nave Bayes liberary

from sklearn.naive_bayes import GaussianNB
# Implementing Naie Bayes

NB_Cls = GaussianNB()

NB_Cls.fit(X_train, y_train)
# Predecting "y" from X test on the basis of the X_train & y__train

y_pred_nb = NB_Cls.predict(X_test)
# Checking Accuracy Score

from sklearn.metrics import accuracy_score

accuracy_score_NB = accuracy_score(y_pred_nb, y_test)

print("Accuracy Score for Naive Bayes Model is", accuracy_score(y_pred_nb,y_test))
# Checking F1 Score

from sklearn.metrics import f1_score

F1_Score_NB = f1_score(y_test,y_pred_nb, average = 'macro')

print("F1 Score for Naive Bayes Model is", f1_score(y_test,y_pred_nb, average='macro'))
# Confusion matrix

from sklearn.metrics import confusion_matrix

confusion_matrix_NB = confusion_matrix(y_test,y_pred_nb)

plt.figure(8);

plt.title('Confusion Matrix - Naive Bayes');

sns.heatmap(confusion_matrix_NB, annot = True);
# Importing SVC from Sklearn's SVM (Support Vector Machine) liberary

from sklearn.svm import SVC
# Implementing SVM

svm_cls = SVC()

svm_cls.fit(X_train,y_train)
# Predecting "y" from X test on the basis of the X_train & y__train

y_pred_svm = svm_cls.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_score_svm = accuracy_score(y_pred_svm, y_test)

print("Accuracy Score for SVM Model is", accuracy_score(y_pred_svm,y_test))
from sklearn.metrics import f1_score

F1_score_svm = f1_score(y_test, y_pred_svm, average = 'macro')

print("F1 Score for SVM Model is", f1_score(y_test,y_pred_svm, average='macro'))
from sklearn.metrics import confusion_matrix

confusion_matrix_svm = confusion_matrix(y_test, y_pred_svm)

plt.figure(9);

sns.heatmap(confusion_matrix_svm, annot = True);

plt.axes().set_title('Confusion Matrix - SVM');
# Importing KNeighborsClassifier from Sklearn's neighbors liberary

from sklearn.neighbors import KNeighborsClassifier
# Implementing KNN

knn_cls = KNeighborsClassifier(n_neighbors=5)

knn_cls.fit(X_train,y_train)
# Predecting "y" from X test on the basis of the X_train & y__train

y_pred_knn = knn_cls.predict(X_test)
# Checking Accuracy Score

from sklearn.metrics import accuracy_score

accuracy_score_knn = accuracy_score(y_pred_knn, y_test)

print("Accuracy Score for K-Nearest Neighbours Model is", accuracy_score(y_pred_knn,y_test))
# Checking F1 Score

from sklearn.metrics import f1_score

F1_score_knn = f1_score(y_test, y_pred_knn, average = 'macro')

print("F1 Score for K-Nearest Neighbours Model is", f1_score(y_test,y_pred_knn, average='macro'))
# Confusion matrix

from sklearn.metrics import confusion_matrix

confusion_matrix_NB = confusion_matrix(y_test,y_pred_knn)

plt.figure(8);

plt.title('Confusion Matrix - K-Nearest Neighbours');

sns.heatmap(confusion_matrix_NB, annot = True);
# In above KNN model we randomly chosed K value = 5.

# Now what if i choose the k value as 2 and i get more accurate model.



F1_Score = []

K_Range = range(1,31)



for K in K_Range:

    knn_clf = KNeighborsClassifier(n_neighbors=K)

    knn_clf.fit(X_train,y_train)

    Y_pred_knn = knn_clf.predict(X_test)

    F1_Score.append(f1_score(Y_pred_knn, y_test, average = 'macro'))

plt.figure(figsize = (10,8));

plt.title("F1 Score Of KNN Model for K Range 1 to 30 - Iris Data");

plt.plot(K_Range, F1_Score, color='green', linestyle='dashed', marker='o',

         markerfacecolor = 'red');

plt.xlabel('K Value');

plt.ylabel('F1 - Score');
# Importing DecisionTreeClassifier from Sklearn's tree liberary

from sklearn.tree import DecisionTreeClassifier
# Implementing Decision Tree Model

dt_cls = DecisionTreeClassifier()

dt_cls.fit(X_train, y_train)
# Predecting "y" from X test on the basis of the X_train & y__train

y_pred_dt = dt_cls.predict(X_test)
# Checking Accuracy Score

from sklearn.metrics import accuracy_score

accuracy_score_dt = accuracy_score(y_test, y_pred_dt)

print("Accuracy Score for Decision Tree Model is", accuracy_score(y_test, y_pred_dt))
# Checking F1 Score

from sklearn.metrics import f1_score

f1_score_dt = f1_score(y_pred_dt, y_test, average = 'macro')

print("F1 Score for Decision Tree Model is", f1_score(y_pred_dt, y_test, average = 'macro'))
# Confusion Matrix

from sklearn.metrics import confusion_matrix

confusion_matrix_dt = confusion_matrix(y_test, y_pred_dt)

plt.figure(11);

sns.heatmap(confusion_matrix_dt, annot = True);

plt.axes().set_title('Confusion Matrix - Decision Tree');
# Creating Data Frame

Score_Matrix = pd.DataFrame({

        "ModelName": ["LogisticRegression", "NaiveBayes", "SVM", "KNN", "DecisionTree"],

        "AccuracyScore": [accuracy_score_LOG, accuracy_score_NB, accuracy_score_svm,

                          accuracy_score_knn, accuracy_score_dt],

        "F1Score": [F1_Score_LOG, F1_Score_NB, F1_score_svm, F1_score_knn, f1_score_dt]        

        })

print(Score_Matrix)
# Plotting score

plt.figure(2)

Score_Matrix.plot.bar(x='ModelName', y = ["F1Score","AccuracyScore"],

                      rot = 0, figsize = (10,7));

plt.title("""Accuracy Score" & "F1 Score" Comparison """);

plt.ylabel("Scores");
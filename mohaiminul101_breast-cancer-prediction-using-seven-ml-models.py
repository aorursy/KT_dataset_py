#Import libraries  

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns
#Load the data 

df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv') 

df.head(10)
#Count the number of rows and columns in the data set

df.shape
#Count the empty (NaN, NAN, na) values in each column

df.isna().sum()
#Drop the column with all missing values (na, NAN, NaN)

#NOTE: This drops the column Unnamed

df = df.dropna(axis=1)
#Get the new count of the number of rows and cols

df.shape
#Get a count of the number of 'M' & 'B' cells

df['diagnosis'].value_counts()
#Visualize this count 

sns.countplot(df['diagnosis'],label="Count")
#Look at the data types 

df.dtypes
#Encoding categorical data values (

from sklearn.preprocessing import LabelEncoder

labelencoder_Y = LabelEncoder()

df.iloc[:,1]= labelencoder_Y.fit_transform(df.iloc[:,1].values)

print(labelencoder_Y.fit_transform(df.iloc[:,1].values))
#Pair Plot

sns.pairplot(df, hue="diagnosis")
# pair plot of sample feature

sns.pairplot(df, hue = 'diagnosis', 

             vars = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean'])
#new data set which now has only 32 columns. Print only the first 5 rows

df.head(5)
#Get the correlation of the columns

df.corr()
#Visualize the correlation by creating a heat map

plt.figure(figsize=(20,20))  

sns.heatmap(df.corr(), annot=True, cmap = 'YlGnBu', fmt='.0%')
#Splitting the data set

X = df.iloc[:, 2:31].values 

Y = df.iloc[:, 1].values 
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
#Function to hold many different models



def models(X_train, Y_train):

    

    #Logistic Regression

    from sklearn.linear_model import LogisticRegression

    log = LogisticRegression(random_state=0)

    log.fit(X_train, Y_train)

    

    #Decision Tree

    from sklearn.tree import DecisionTreeClassifier

    tree = DecisionTreeClassifier(criterion = 'entropy', random_state=0)

    tree.fit(X_train, Y_train)

    

    #Random Forest Classifier

    from sklearn.ensemble import RandomForestClassifier

    forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

    forest.fit(X_train, Y_train)

    

    #Support Vector Machines

    from sklearn.svm import SVC

    sv = SVC(random_state=0)

    sv.fit(X_train, Y_train)

    

    #K-Nearest Neighbors 

    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=5)

    knn.fit(X_train, Y_train)

    

    #Naive Bayes Classifier

    from sklearn.naive_bayes import GaussianNB

    gnb = GaussianNB ()

    gnb.fit(X_train, Y_train)

    

    #AdaBoost Classifier

    from sklearn.ensemble import AdaBoostClassifier

    clf = AdaBoostClassifier()

    clf.fit(X_train, Y_train)

    

    #Print the models accuracy on the training data

    print('[0]Logistic Regression Classifier Training Accuracy:', log.score(X_train, Y_train))

    print('[1]Decision Tree Classifier Training Accuracy:', tree.score(X_train, Y_train))

    print('[2]Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train))

    print('[3]Support Vector Machines Classifier Training Accuracy:', sv.score(X_train, Y_train))

    print('[4]K-Nearest Neighbors Classifier Training Accuracy:', knn.score(X_train, Y_train))

    print('[5]Naive Bayes Classifier Classifier Training Accuracy:', gnb.score(X_train, Y_train))

    print('[6]AdaBoost Classifier Training Accuracy:', clf.score(X_train, Y_train))

    

    return log, tree, forest, sv, knn, gnb, clf
#Getting all of the models

model = models(X_train,Y_train)
#confusion matrix and the accuracy



from sklearn.metrics import confusion_matrix

for i in range(len(model)):

  cm = confusion_matrix(Y_test, model[i].predict(X_test))

  

  TN = cm[0][0]

  TP = cm[1][1]

  FN = cm[1][0]

  FP = cm[0][1]

  

  print(cm)

  print('Model[{}] Testing Accuracy = "{}!"'.format(i,  (TP + TN) / (TP + TN + FN + FP)))

  print()# Print a new line
#Show other ways to get the classification accuracy & other metrics 



from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score



for i in range(len(model)):

  print('Model ',i)

  #Check precision, recall, f1-score

  print( classification_report(Y_test, model[i].predict(X_test)) )

  #Another way to get the models accuracy on the test data

  print( accuracy_score(Y_test, model[i].predict(X_test)))

  print()#Print a new line
# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import warnings 

warnings.filterwarnings('ignore')
# Importing the dataset

dataset = pd.read_csv('../input/heart-disease-uci/heart.csv')
dataset.head()
dataset.info()
# Checking for null values in the dataset

dataset.isnull().values.any()
# Visualising the dataset

plt.figure(figsize=(9,7))

plt.style.use('seaborn-pastel')

labels=['female','male']

sns.set_style("darkgrid")

ax=sns.barplot(x='target',y='age',data=dataset,hue='sex')

h, l = ax.get_legend_handles_labels()

ax.legend(h,labels,title="Gender",loc='upper right')

ax.set_ylabel("Age",fontdict={'fontsize' : 12})

ax.set_xlabel("Target variable: Angiographic disease status",fontdict={'fontsize' : 12})

for p in ax.patches:

    width, height = p.get_width(), p.get_height()

    x, y = p.get_xy() 

    ax.annotate('{:.2f}'.format(height), (x+ 0.15, y + height + 2.4))

plt.title('Mean age of patients grouped by gender',fontweight="bold")

plt.show()
# Defining the features and the outcome variable

x= dataset.iloc[:,:-1].values

y = dataset.iloc[:,-1].values
# Splitting the dataset into training set and test set

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
## Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train[:,[0,3,4,7,9,11]] = sc.fit_transform(x_train[:,[0,3,4,7,9,11]])

x_test[:,[0,3,4,7,9,11]] = sc.transform(x_test[:,[0,3,4,7,9,11]])
print(x_train[0])
print(x_test[0])
## Applying the Logistic regression model on the training set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)

classifier.fit(x_train,y_train)
## Predicting test results

y_pred = classifier.predict(x_test)

np.set_printoptions(precision=2)

print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
# Visualising the test results vs predicted results

bins = np.linspace(-1,2,10)

plt.figure(figsize=(8,6))

ax =plt.hist([y_test,y_pred],bins=bins,color=['blue','lawngreen'],label=['Actual result','Predicted result'],align='left')

plt.xlabel('Target variable: Angiographic disease status',fontdict={'fontsize' : 12})

plt.ylabel('Test set sample size',fontdict={'fontsize' : 12})

plt.xlim(-1,2)

plt.xticks([0,1])

plt.ylim(0,len(y_test))

plt.legend(prop={'size': 12})

plt.show()
## Calculating the accuracy score and confusion matrix

from sklearn.metrics import accuracy_score,confusion_matrix

ac_LogReg = accuracy_score(y_test,y_pred)

cm = confusion_matrix(y_test,y_pred)
## Printing the accuracy score

print(ac_LogReg)
## Printing the confusion matrix

print(cm)
## Applying the non-linear SVC model on the training set

from sklearn.svm import SVC

classifier = SVC(kernel='rbf',random_state=0)

classifier.fit(x_train,y_train)
## Predicting test results

y_pred = classifier.predict(x_test)

np.set_printoptions(precision=2)

print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
# Visualising the test results vs predicted results

bins = np.linspace(-1,2,10)

plt.figure(figsize=(8,6))

ax =plt.hist([y_test,y_pred],bins=bins,color=['blue','lawngreen'],label=['Actual result','Predicted result'],align='left')

plt.xlabel('Target variable: Angiographic disease status',fontdict={'fontsize' : 12})

plt.ylabel('Test set sample size',fontdict={'fontsize' : 12})

plt.xlim(-1,2)

plt.xticks([0,1])

plt.ylim(0,len(y_test))

plt.legend(prop={'size': 12})

plt.show()
## Calculating the accuracy score and confusion matrix

from sklearn.metrics import accuracy_score,confusion_matrix

ac_SVC = accuracy_score(y_test,y_pred)

cm = confusion_matrix(y_test,y_pred)
## Printing the accuracy score

print(ac_SVC)
## Printing the confusion matrix

print(cm)
## Applying the Linear SVC model on the training set

from sklearn.svm import SVC

classifier = SVC(kernel='linear',random_state=0)

classifier.fit(x_train,y_train)
## Predicting test results

y_pred = classifier.predict(x_test)

np.set_printoptions(precision=2)

print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
# Visualising the test results vs predicted results

bins = np.linspace(-1,2,10)

plt.figure(figsize=(8,6))

ax =plt.hist([y_test,y_pred],bins=bins,color=['blue','lawngreen'],label=['Actual result','Predicted result'],align='left')

plt.xlabel('Target variable: Angiographic disease status',fontdict={'fontsize' : 12})

plt.ylabel('Test set sample size',fontdict={'fontsize' : 12})

plt.xlim(-1,2)

plt.xticks([0,1])

plt.ylim(0,len(y_test))

plt.legend(prop={'size': 12})

plt.show()
## Calculating the accuracy score and confusion matrix

from sklearn.metrics import accuracy_score,confusion_matrix

ac_LinearSVC = accuracy_score(y_test,y_pred)

cm = confusion_matrix(y_test,y_pred)
## Printing accuracy score

print(ac_LinearSVC)
#Printing confusion matrix

print(cm)
## Applying the K-NN classification model on the training set

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()

classifier.fit(x_train,y_train)
## Predicting test results

y_pred = classifier.predict(x_test)

np.set_printoptions(precision=2)

print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
# Visualising the test results vs predicted results

bins = np.linspace(-1,2,10)

plt.figure(figsize=(8,6))

ax =plt.hist([y_test,y_pred],bins=bins,color=['blue','lawngreen'],label=['Actual result','Predicted result'],align='left')

plt.xlabel('Target variable: Angiographic disease status',fontdict={'fontsize' : 12})

plt.ylabel('Test set sample size',fontdict={'fontsize' : 12})

plt.xlim(-1,2)

plt.xticks([0,1])

plt.ylim(0,len(y_test))

plt.legend(prop={'size': 12})

plt.show()
## Calculating the accuracy score and confusion matrix

from sklearn.metrics import accuracy_score,confusion_matrix

ac_KNN = accuracy_score(y_test,y_pred)

cm = confusion_matrix(y_test,y_pred)
# Printing accuarcy score

print(ac_KNN)
# Printing confusion matrix

print(cm)
## Applying the Naive Bayes classification model on the training set

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(x_train,y_train)
## Predicting test results

y_pred = classifier.predict(x_test)

np.set_printoptions(precision=2)

print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
# Visualising the test results vs predicted results

bins = np.linspace(-1,2,10)

plt.figure(figsize=(8,6))

ax =plt.hist([y_test,y_pred],bins=bins,color=['blue','lawngreen'],label=['Actual result','Predicted result'],align='left')

plt.xlabel('Target variable: Angiographic disease status',fontdict={'fontsize' : 12})

plt.ylabel('Test set sample size',fontdict={'fontsize' : 12})

plt.xlim(-1,2)

plt.xticks([0,1])

plt.ylim(0,len(y_test))

plt.legend(prop={'size': 12})

plt.show()
## Calculating the accuracy score and confusion matrix

from sklearn.metrics import accuracy_score,confusion_matrix

ac_NBayes = accuracy_score(y_test,y_pred)

cm = confusion_matrix(y_test,y_pred)
# Printing accuracy score

print(ac_NBayes)
# Printing confusion matrix

print(cm)
## Applying the Decision Tree Classification model on the training set

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)

classifier.fit(x_train,y_train)
## Predicting test results

y_pred = classifier.predict(x_test)

np.set_printoptions(precision=2)

print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
# Visualising the test results vs predicted results

bins = np.linspace(-1,2,10)

plt.figure(figsize=(8,6))

ax =plt.hist([y_test,y_pred],bins=bins,color=['blue','lawngreen'],label=['Actual result','Predicted result'],align='left')

plt.xlabel('Target variable: Angiographic disease status',fontdict={'fontsize' : 12})

plt.ylabel('Test set sample size',fontdict={'fontsize' : 12})

plt.xlim(-1,2)

plt.xticks([0,1])

plt.ylim(0,len(y_test))

plt.legend(prop={'size': 12})

plt.show()
## Calculating the accuracy score and confusion matrix

from sklearn.metrics import accuracy_score,confusion_matrix

ac_DecisionTree = accuracy_score(y_test,y_pred)

cm = confusion_matrix(y_test,y_pred)
## Printing accuracy score

print(ac_DecisionTree)
## Printing confusion matrix

print(cm)
## Applying the Random Forest Classification model on the training set

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=200,criterion='entropy',random_state=0)

classifier.fit(x_train,y_train)
## Predicting test results

y_pred = classifier.predict(x_test)

np.set_printoptions(precision=2)

print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
# Visualising the test results vs predicted results

bins = np.linspace(-1,2,10)

plt.figure(figsize=(8,6))

arr =plt.hist(x=[y_test,y_pred],bins=bins,color=['blue','lawngreen'],label=['Actual result','Predicted result'],align='left')

plt.xlabel('Target variable: Angiographic disease status',fontdict={'fontsize' : 12})

plt.ylabel('Test set sample size',fontdict={'fontsize' : 12})

plt.xlim(-1,2)

plt.xticks([0,1])

plt.ylim(0,len(y_test))

plt.legend(prop={'size': 12})

plt.show()
## Calculating the accuracy score and confusion matrix

from sklearn.metrics import accuracy_score,confusion_matrix

ac_randomForest = accuracy_score(y_test,y_pred)

cm = confusion_matrix(y_test,y_pred)
## Printing accuarcy score

print(ac_randomForest)
## Printing confusion matrix

print(cm)
## Applying the XGBoost Classification model on the training set

from xgboost import XGBClassifier

classifier = XGBClassifier()

classifier.fit(x_train,y_train)
## Predicting test results

y_pred = classifier.predict(x_test)

np.set_printoptions(precision=2)

print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
# Visualising the test results vs predicted results

bins = np.linspace(-1,2,10)

plt.figure(figsize=(8,6))

arr =plt.hist(x=[y_test,y_pred],bins=bins,color=['blue','lawngreen'],label=['Actual result','Predicted result'],align='left')

plt.xlabel('Target variable: Angiographic disease status',fontdict={'fontsize' : 12})

plt.ylabel('Test set sample size',fontdict={'fontsize' : 12})

plt.xlim(-1,2)

plt.xticks([0,1])

plt.ylim(0,len(y_test))

plt.legend(prop={'size': 12})

plt.show()
## Calculating the accuracy score and confusion matrix

from sklearn.metrics import accuracy_score,confusion_matrix

ac_xgb = accuracy_score(y_test,y_pred)

cm = confusion_matrix(y_test,y_pred)
## Printing accuarcy score

print(ac_xgb)
## Printing confusion matrix

print(cm)
#Comparing results of the models based on accuracy score

modelList = [ac_LogReg,ac_SVC,ac_LinearSVC,ac_KNN,ac_NBayes,ac_DecisionTree,ac_randomForest,ac_xgb]
for i in range(0,len(modelList)):

    modelList[i] = modelList[i]*100

print(modelList)
labelList = ['Logistic Regression','Kernel SVC','Linear SVC','K-NN','Naive Bayes','Decision Tree','Random Forest',

            'XGBoost']

plt.figure(figsize=(12,7))

sns.set_style('dark')

ax = sns.barplot(x=labelList,y=modelList,palette=sns.cubehelix_palette(8))

plt.ylim(0,100)

plt.title('Accuracy score comparison among different classification models')

for p in ax.patches:

    width, height = p.get_width(), p.get_height()

    x, y = p.get_xy() 

    ax.annotate('{:.2f}%'.format(height), (x+0.25, y + height + 0.8))

plt.show()
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#Read the train dataset
df = pd.read_csv('/kaggle/input/mobile-price-classification/train.csv')
#first few rows
df.head()
#length of dataset
len(df)
#check for null values
df.isnull().sum()
#confirm with heatmap
sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
#check the datatypes
df.dtypes
#report
df.describe().T
sns.catplot(x = 'blue', data = df, kind = 'count')
sns.catplot(x = 'four_g', data = df, kind = 'count')
sns.catplot(x = 'three_g', data = df, kind = 'count')
sns.catplot(x = 'touch_screen', data = df, kind = 'count')
sns.catplot(x = 'price_range', data = df, kind = 'count')
#number of 3G phones with respect to number of cores
import matplotlib.pyplot as plt
plt.figure(figsize = (10,6))
sns.countplot(x = "n_cores", hue = 'three_g', data = df)
plt.title("3G with respect to number of cores")
plt.show()
#number of 4G phones with respect to number of cores
plt.figure(figsize = (10,6))
sns.countplot(x = "n_cores", hue = 'four_g', data = df)
plt.title("4G with respect to number of cores")
plt.show()
#number of touch screen phones with respect to price range
plt.figure(figsize = (10,6))
sns.countplot(x = "price_range", hue = 'touch_screen', data = df)
plt.title("Count of touch screen phones for each price range")
plt.show()
#number of dual sim phones for each price range
plt.figure(figsize = (10,6))
sns.countplot(x = "price_range", hue = 'dual_sim', data = df)
plt.title("Count of dual sim phones for each price range")
plt.show()
#Choose price range as the desired label. Check correlation of the table.
df.corr()['price_range'].sort_values()
#heatmap for correlation
f, ax = plt.subplots(figsize = (20,12))
sns.heatmap(df.corr(), vmax = 0.8, square = True)
#Divide the dataset into X and y
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
#scale the value of X
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
#split into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, multi_class = 'auto', solver = 'lbfgs')
classifier.fit(X_train, y_train)
#predicting the values
y_pred = classifier.predict(X_test)
#confusion matrix
from sklearn.metrics import confusion_matrix
cm_Log = confusion_matrix(y_test, y_pred)
cm_Log
#Accuracy and report of the classifier
from sklearn.metrics import accuracy_score, classification_report
accuracy_score(y_test,y_pred)
#report
report_Log = classification_report(y_test, y_pred)
print(report_Log)
#KNNeighbours
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p =2)
classifier.fit(X_train, y_train)
#predicting the values
y_pred = classifier.predict(X_test)
#confusion matrix
cm_KNN = confusion_matrix(y_test, y_pred)
print(cm_KNN)
#Accuracy
accuracy_score(y_test,y_pred)
#Report
report_KNN = classification_report(y_test, y_pred)
print(report_KNN)
#SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0, C = 100)
classifier.fit(X_train,y_train)
#predicting the values
y_pred = classifier.predict(X_test)
#confusion matrix
cm_SVC = confusion_matrix(y_test, y_pred)
print(cm_SVC)
#Accuracy
accuracy_score(y_test,y_pred)
#Report
report_SVC = classification_report(y_test, y_pred)
print(report_SVC)
#NaiveBayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)
#predicting the values
y_pred = classifier.predict(X_test)
#confusion matrix
cm_NB = confusion_matrix(y_test, y_pred)
print(cm_NB)
#Accuracy
accuracy_score(y_test,y_pred)
#Report
report_NB = classification_report(y_test, y_pred)
print(report_NB)
#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train,y_train)
#predicting the values
y_pred = classifier.predict(X_test)
#confusion matrix
cm_RF = confusion_matrix(y_test, y_pred)
print(cm_RF)
#Accuracy
accuracy_score(y_test,y_pred)
#Report
report_RF = classification_report(y_test, y_pred)
print(report_RF)
#Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
classifier = GradientBoostingClassifier()
classifier.fit(X_train,y_train)
#predicting the values
y_pred = classifier.predict(X_test)
#confusion matrix
cm_GBC = confusion_matrix(y_test, y_pred)
print(cm_GBC)
#Accuracy
accuracy_score(y_test,y_pred)
#Report
report_GBC = classification_report(y_test, y_pred)
print(report_GBC)
#Select SVM for classification
Test = pd.read_csv('/kaggle/input/mobile-price-classification/test.csv')
df_test = Test.drop(['id'], axis = 1)
df_test = sc.fit_transform(df_test)
#classify
y_Test_values = classifier.predict(df_test)
#join the predicted values to the test dataset
Test['price_range'] = y_Test_values
#display Test 
Test.head()
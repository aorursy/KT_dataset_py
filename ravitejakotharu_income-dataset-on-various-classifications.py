# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing libaries for various classification techniques

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
#from pandas.plotiing import scatter_matrix
from sklearn.linear_model import LogisticRegression # logistic regression
from sklearn import svm # support vector machine
from sklearn.ensemble import RandomForestClassifier #Random_forest
from sklearn.tree import DecisionTreeClassifier #Decision tree
from sklearn.naive_bayes import GaussianNB #Naive_bayes
from sklearn.neighbors import KNeighborsClassifier #K nearest neighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#Reading the data set
df = pd.read_csv('/kaggle/input/income-dataset/income_data1.csv')
df.info()
df.head()
#Replacing the some special character columns names with proper names 

df.rename(columns={'capital-gain': 'capital gain', 'capital-loss': 'capital loss', 'native-country': 'country','hours-per-week': 'hours per week','marital-status': 'marital'}, inplace=True)
df.columns
#Finding the special characters in the data frame
df.isin(['?']).sum(axis=0)
#Replacing the special character to nan and then drop the columns
df['country'] = df['country'].replace('?',np.nan)
df['workclass'] = df['workclass'].replace('?',np.nan)
df['occupation'] = df['occupation'].replace('?',np.nan)
#Dropping the NaN rows now 
df.dropna(how='any',inplace=True)
#Assigning the numeric values to the string type variables
number = LabelEncoder()
df['workclass'] = number.fit_transform(df['workclass'])
df['education'] = number.fit_transform(df['education'])
df['marital'] = number.fit_transform(df['marital'])
df['occupation'] = number.fit_transform(df['occupation'])
df['relationship'] = number.fit_transform(df['relationship'])
df['race'] = number.fit_transform(df['race'])
df['gender'] = number.fit_transform(df['gender'])
df['country'] = number.fit_transform(df['country'])
df['income'] = number.fit_transform(df['income'])
df.head(5)
#Here we were grouping the each columns with prefernce to the income set
df.groupby('education').income.mean().plot(kind='bar')
df.groupby('workclass').income.mean().plot(kind='bar')
df.groupby('gender').income.mean().plot(kind='bar')
df.groupby('race').income.mean().plot(kind='bar')
df.groupby('relationship').income.mean().plot(kind='bar')

df.groupby('occupation').income.mean().plot(kind='bar')

df.groupby('marital').income.mean().plot(kind='bar')
# summarize the class distribution
target = df.values[:,-1]
counter = Counter(target)
for k,v in counter.items():
	per = v / len(target) * 100
	print('Class=%s, Count=%d, Percentage=%.3f%%' % (k, v, per))
#Train_Test splitting
X = df.drop(['income'],axis=1)
y = df['income']
X.head()
y.head()
#Declaring the train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4,random_state=0)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X.columns)

X_test = pd.DataFrame(scaler.transform(X_test), columns = X.columns)
X_train.head()
# Logistic Regression
LR = LogisticRegression()
LR.fit(X_train, y_train)
y_pred = LR.predict(X_test)
score_LR = LR.score(X_test,y_test)
print('The accuracy of the Logistic Regression model is', score_LR)
targets = ['<=50k' , '>50k']
print(classification_report(y_test, y_pred,target_names=targets))
# Support Vector Classifier (SVM/SVC)
from sklearn.svm import SVC
svc = SVC(gamma=0.22)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
score_svc = svc.score(X_test,y_test)
print('The accuracy of SVC model is', score_svc)
targets = ['<=50k' , '>50k']
print(classification_report(y_test, y_pred,target_names=targets))
# Random Forest Classifier
RF = RandomForestClassifier()
RF.fit(X_train, y_train)
y_pred = RF.predict(X_test)
score_RF = RF.score(X_test,y_test)
print('The accuracy of the Random Forest Model is', score_RF)
targets = ['<=50k' , '>50k']
print(classification_report(y_test, y_pred,target_names=targets))
# Decision Tree
DT = DecisionTreeClassifier()
DT.fit(X_train,y_train)
y_pred = DT.predict(X_test)
score_DT = DT.score(X_test,y_test)
print("The accuracy of the Decision tree model is ",score_DT)
targets = ['<=50k' , '>50k']
print(classification_report(y_test, y_pred,target_names=targets))
# Gaussian Naive Bayes
GNB = GaussianNB()
GNB.fit(X_train, y_train)
y_pred = GNB.predict(X_test)
score_GNB = GNB.score(X_test,y_test)
print('The accuracy of Gaussian Naive Bayes model is', score_GNB)
targets = ['<=50k' , '>50k']
print(classification_report(y_test, y_pred,target_names=targets))
# K-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
score_knn = knn.score(X_test,y_test)
print('The accuracy of the KNN Model is',score_knn)
targets = ['<=50k' , '>50k']
print(classification_report(y_test, y_pred,target_names=targets))
tabular_form = {'CLASSIFICATION':['LogisticRegression','SupportVectorClassifier','RandomForestClassifier','DecisionTree','GaussianNaiveBayes','K-NearestNeighbors'],
                'ACCURACY':[score_LR,score_svc,score_RF,score_DT,score_GNB,score_knn]
                }
tf = pd.DataFrame(tabular_form,columns= ['CLASSIFICATION','ACCURACY'])
print(tf)
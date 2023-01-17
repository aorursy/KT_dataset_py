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
data=pd.read_csv('/kaggle/input/heart-disease-prediction-using-logistic-regression/framingham.csv')

data.head()
data.describe()
data.info()


X = data.drop(['TenYearCHD'], axis=1, inplace=False)

print('X Data is \n' , X.head())

print('X shape is ' , X.shape)



y = data['TenYearCHD']

print('y Data is \n' , y.head())

print('y shape is ' , y.shape)
X = X.apply(lambda x: x.fillna(x.mean()),axis=0)

X.isnull().sum(axis = 0)
from sklearn.model_selection import train_test_split



#Splitting data



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)



#Splitted Data

print('X_train shape is ' , X_train.shape)

print('X_test shape is ' , X_test.shape)

print('y_train shape is ' , y_train.shape)

print('y_test shape is ' , y_test.shape)
from sklearn.linear_model import LogisticRegression

LogisticRegressionModel = LogisticRegression(penalty='l2',solver='sag',C=1.0,random_state=33)

LogisticRegressionModel.fit(X_train, y_train)

y_pred = LogisticRegressionModel.predict(X_test)
print('LogisticRegressionModel Train Score is : ' , LogisticRegressionModel.score(X_train, y_train))

print('LogisticRegressionModel Test Score is : ' , LogisticRegressionModel.score(X_test, y_test))
from sklearn.metrics import classification_report





ClassificationReport = classification_report(y_test,y_pred)

print('Classification Report is : ', ClassificationReport )
from sklearn.metrics import confusion_matrix

import seaborn as sns

import matplotlib.pyplot as plt



CM = confusion_matrix(y_test, y_pred)

print('Confusion Matrix is : \n', CM)

sns.heatmap(CM, center = True)

plt.show()
from sklearn.svm import SVC







SVCModel = SVC(kernel= 'rbf',# it can be also linear,poly,sigmoid,precomputed

               max_iter=100,C=1.0,gamma='auto')

SVCModel.fit(X_train, y_train)



print('SVCModel Train Score is : ' , SVCModel.score(X_train, y_train))

print('SVCModel Test Score is : ' , SVCModel.score(X_test, y_test))

y_pred = SVCModel.predict(X_test)





ClassificationReport = classification_report(y_test,y_pred)

print('Classification Report is : ', ClassificationReport )
CM = confusion_matrix(y_test, y_pred)

print('Confusion Matrix is : \n', CM)

sns.heatmap(CM,center = True)

plt.show()
from sklearn.tree import DecisionTreeClassifier



DecisionTreeClassifierModel = DecisionTreeClassifier(criterion='gini',max_depth=3,random_state=33) #criterion can be entropy

DecisionTreeClassifierModel.fit(X_train, y_train)

y_pred = DecisionTreeClassifierModel.predict(X_test)

print('DecisionTreeClassifierModel Train Score is : ' , DecisionTreeClassifierModel.score(X_train, y_train))

print('DecisionTreeClassifierModel Test Score is : ' , DecisionTreeClassifierModel.score(X_test, y_test))
print('DecisionTreeClassifierModel feature importances are : ' , DecisionTreeClassifierModel.feature_importances_)
ClassificationReport = classification_report(y_test,y_pred)

print('Classification Report is : ', ClassificationReport )
CM = confusion_matrix(y_test, y_pred)

print('Confusion Matrix is : \n', CM)

sns.heatmap(CM,center = True)

plt.show()
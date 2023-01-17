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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from  matplotlib.pyplot import subplot
%matplotlib inline

from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,mean_squared_error
red_wine = pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
#Let's check how the data is distributed
red_wine.head()
#Information about the data columns
red_wine.info()
red_wine.describe()
red_wine.quality.value_counts()
red_wine.isnull().sum()
#fixed acidity does not give any specification to classify the quality.
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'fixed acidity', data = red_wine)
#Relationship between each variables
plt.figure(figsize=(20,10))
sns.heatmap(red_wine.corr(), annot=True,cmap='Reds')
plt.show()
#a downing trend in the volatile acidity as the quality goes higher 
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'volatile acidity', data = red_wine)
#pie plot showing quality
plt.figure(1, figsize=(8,8))
red_wine['quality'].value_counts().plot.pie(autopct="%1.1f%%")
#histogram
sns.countplot(red_wine['quality'])
#Composition of citric acid go higher as quality of the wine goes higher
fig = plt.figure(figsize = (10,6))
sns.violinplot(x = 'quality', y = 'citric acid', data = red_wine)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'residual sugar', data = red_wine)
#Composition of chloride goes down as quality of the wine goes higher
fig = plt.figure(figsize = (10,6))
sns.boxenplot(x = 'quality', y = 'chlorides', data = red_wine)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = red_wine)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'total sulfur dioxide', data = red_wine)
#Sulphates level goes higher with the quality of wine
fig = plt.figure(figsize = (10,6))
sns.violinplot(x = 'quality', y = 'sulphates', data = red_wine)
#Alcohol level goes higher as quality of wine increases
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'alcohol', data = red_wine)
#Making binary classificaion for the response variable.
#Dividing wine as good and bad by giving the limit for the quality
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
red_wine['quality'] = pd.cut(red_wine['quality'], bins = bins, labels = group_names)
#Now lets assign a labels to our quality variable
label_quality = LabelEncoder()
#Bad becomes 0 and good becomes 1 
red_wine['quality'] = label_quality.fit_transform(red_wine['quality'])
red_wine['quality'].value_counts()
sns.countplot(red_wine['quality'])
#Now seperate the dataset as response variable and feature variabes
X = red_wine.drop('quality', axis = 1)
y = red_wine['quality']
#Train and Test splitting of data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
#Applying Standard scaling to get optimized result
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
#Let's see how our model performed
print(classification_report(y_test, pred_rfc))
print(accuracy_score(y_test,pred_rfc))
svm = SVC()
svm.fit(X_train, y_train)
pred_svm = svm.predict(X_test)
print(classification_report(y_test, pred_svm))
print(accuracy_score(y_test,pred_svm))
sgd = SGDClassifier(penalty=None)
sgd.fit(X_train, y_train)
pred_sgd = sgd.predict(X_test)
print(classification_report(y_test, pred_sgd))
print(accuracy_score(y_test,pred_sgd))
xgb = XGBClassifier(max_depth=3,n_estimators=200,learning_rate=0.5)
xgb.fit(X_train,y_train)
pred_xgb = xgb.predict(X_test)
print(classification_report(y_test, pred_xgb))
print(accuracy_score(y_test,pred_xgb))
#for SGD
print(confusion_matrix(y_test, pred_sgd))
#for randomforest
print(confusion_matrix(y_test, pred_rfc))
#for SVM
print(confusion_matrix(y_test, pred_svm))
#for XGB
print(confusion_matrix(y_test, pred_xgb))
x=red_wine.drop('quality', axis = 1)
y= red_wine['quality']
scaler = MinMaxScaler(feature_range = (0,1))
scaler.fit_transform(x)

x.head()
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)
classifier_log = LogisticRegression()
model = classifier_log.fit(x_train,y_train)

y_pred_log = classifier_log.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred_log, y_test)*100)
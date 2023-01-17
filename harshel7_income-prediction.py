# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, chi2
#Analyse the first few rows of the dataset
df = pd.read_csv('../input/adult.csv')
df.head()
#Checking the shape of the dataset 
df.shape

#nrows = 48842 and ncols = 15
#describe method provides the basic info about the dataset such as max, min value. Standard deviation,
#mean, median etc
df.describe()
#info method is another useful feature that can be used to check if any missing values are present
# in our feature columns. It also gives the data types of our feature columns
df.info()
#Check the percentage and number of missing values in feature columns

for i in df.columns:
    non_value = df[i].isin(['?']).sum()
    if non_value > 0:
        print(i)
        print('{}'.format(float(non_value) / (df[i].shape[0]) * 100))
        print('\n')
#selecting all the rows without the '?' sign.
df = df[df['workclass'] != '?']
df = df[df['occupation'] != '?']
df = df[df['native-country'] != '?']
# This 'fnlwgt' feature does'nt seem to make any sense and also the mean value of this feature is too high we can
#remove it
df = df.drop('fnlwgt', axis=1)
plt.figure(figsize=(12,12))
sns.heatmap(df.corr(), annot=True, cmap='magma', linecolor='white', linewidths=1)
plt.show()
plt.figure(figsize=(12,8))
sns.countplot(df['income'], hue = df['education'], palette = 'rainbow', edgecolor = [(0,0,0), (0,0,0)])
plt.show()
plt.figure(figsize=(12,8))
sns.countplot(y = df['income'], hue = df['gender'], palette = 'summer', edgecolor = [(0,0,0), (0,0,0)])
plt.show()
print("The number of men with each qualification")
print(df[df['gender'] == 'Male']['education'].value_counts())
plt.figure(figsize=(12,8))
sns.barplot(x = df[df['gender'] == 'Male']['education'].value_counts().values, y = df[df['gender'] == 'Male']['education'].value_counts().index, data = df)
plt.show()
("The number of women with each qualification")
print(df[df['gender'] == 'Female']['education'].value_counts())
plt.figure(figsize=(12,8))
sns.barplot(x = df[df['gender'] == 'Female']['education'].value_counts().values, y = df[df['gender'] == 'Female']['education'].value_counts().index, data = df)
plt.show()
plt.figure(figsize=(12,8))
sns.countplot(df['income'], hue = df['relationship'], palette = 'autumn', edgecolor = [(0,0,0), (0,0,0)])
plt.show()
plt.figure(figsize=(12,8))
sns.countplot(df['income'], hue = df['occupation'], palette = 'BuGn_d', edgecolor = [(0,0,0), (0,0,0)])
plt.show()
plt.figure(figsize=(12,8))
sns.countplot(df['income'], hue = df['race'], palette = 'Set3', edgecolor = [(0,0,0), (0,0,0)])
plt.show()
plt.figure(figsize=(12,8))
sns.countplot(df['income'], hue = df['workclass'], palette = 'Dark2', edgecolor = [(0,0,0), (0,0,0)])
plt.show()
plt.figure(figsize=(12, 10))
sns.boxplot(x='income', y='age', data=df, hue='gender', palette = 'prism')
plt.show()
#Lets check the unique variables of the feature
df['marital-status'].unique()
#Replace the unwanted variables and distribute the variables into two variables namely 'married' and 'not married'

df['marital-status'] = df['marital-status'].replace(['Never-married', 'Married-civ-spouse', 'Widowed', 'Separated', 'Divorced',
                                  'Married-spouse-absent', 'Married-AF-spouse'], ['not married', 'married', 'not married',
                                   'not married', 'not married', 'not married', 'married'])
#Lets chech the head again of the dataframe
df.head()
df = pd.get_dummies(df, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship','race', 'gender',
                           'native-country'], drop_first=True)
df.head()
# Split the dataframe into features (X) and labels(y)
X = df.drop('income', axis=1)
y = df['income']
y = pd.get_dummies(y, columns=y, drop_first=True)
y = y.iloc[:,-1]
y.shape
# Split the dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)
feature_select = SelectKBest(chi2, k = 8)  #finding the top 8 best features
feature_select.fit(X_train, y_train)
score_list = feature_select.scores_
top_features = X_train.columns
uni_features = list(zip(score_list, top_features))
print(sorted(uni_features, reverse=True)[0:8])
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

X_train_1 = feature_select.transform(X_train)
X_test_1 = feature_select.transform(X_test)

#random forest classifier with n_estimators=10 (default)
rf_clf = RandomForestClassifier()      
rf_clf.fit(X_train_1,y_train)

rf_pred = rf_clf.predict(X_test_1)

accu_rf = accuracy_score(y_test, rf_pred)
print('Accuracy is: ',accu_rf)

cm_1 = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm_1, annot=True, fmt="d")
plt.show()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

X_train_2 = feature_select.transform(X_train)
X_test_2 = feature_select.transform(X_test)


knn_clf = KNeighborsClassifier(n_neighbors=1)      
knn_clf.fit(X_train_2,y_train)

knn_pred = knn_clf.predict(X_test_2)

accu_knn = accuracy_score(y_test, knn_pred)
print('Accuracy is: ',accu_knn)

cm_2 = confusion_matrix(y_test, knn_pred)
sns.heatmap(cm_2, annot=True, fmt="d")
plt.show()
accu_score = []

for k in range(1, 50):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train_2, y_train)
    prediction = knn.predict(X_test_2)
    accu_score.append(accuracy_score(prediction, y_test))
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.plot(range(1, 50), accu_score)
plt.xlabel('K values')
plt.ylabel('Accuracy score')
plt.show()
X_train_3 = feature_select.transform(X_train)
X_test_3 = feature_select.transform(X_test)


knn_clf_1 = KNeighborsClassifier(n_neighbors=28)      
knn_clf_1.fit(X_train_2,y_train)

knn_pred_1 = knn_clf_1.predict(X_test_2)

accu_knn_1 = accuracy_score(y_test, knn_pred_1)
print('Accuracy is: ',accu_knn_1)

cm_3 = confusion_matrix(y_test, knn_pred_1)
sns.heatmap(cm_3, annot=True, fmt="d", cmap='Dark2')
plt.show()
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

kfold = KFold(n_splits = 10, random_state = 5)

result = cross_val_score(rf, X_train_1, y_train, cv=kfold, scoring='accuracy')

print(result.mean())
from sklearn.feature_selection import RFECV

# The "accuracy" scoring is proportional to the number of correct classifications
clf_rf_3 = RandomForestClassifier() 
rfecv = RFECV(estimator=clf_rf_3, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation
rfecv = rfecv.fit(X_train, y_train)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', X_train.columns[rfecv.support_])
import matplotlib.pyplot as plt

plt.figure(figsize = (10,8))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
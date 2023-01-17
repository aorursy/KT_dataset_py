# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Loading dataset
df = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
df.head()
df.isnull().values.any()
#Data visualization 
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),annot=True,linewidth = 1, cmap="rocket_r")
plt.title("Heatmap Correlation of the Dataset", fontsize = 20)
#Making binary classificaion for the response variable.
#Dividing wine as good and bad by giving the limit for the quality
bins = (2, 6.5, 8)
labels = ['bad', 'good']
df['quality'] = pd.cut(x = df['quality'], bins = bins, labels = labels)
df['quality'].value_counts()
#Assigning 0 to bad and 1 to good quality
le = LabelEncoder()
df['quality'] = le.fit_transform(df['quality'])
df['quality'].value_counts()
#Splitting and scaling the data
X = df.drop('quality', axis = 1)
sc = StandardScaler()
X = sc.fit(X).transform(X)

y = df['quality']

#Train and Test splitting 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
lr = LogisticRegression(C=0.8, solver='liblinear', fit_intercept=True, penalty = 'l1')
lr.fit(X_train, y_train)
cv_lr = cross_val_score(estimator = lr, X = X_train, y = y_train.ravel(), cv = 10)
print("CV: ", cv_lr.mean())

y_pred_lr_train = lr.predict(X_train)
accuracy_lr_train = accuracy_score(y_train, y_pred_lr_train)
print("Training set: ", accuracy_lr_train)

y_pred_lr_test = lr.predict(X_test)
accuracy_lr_test = accuracy_score(y_test, y_pred_lr_test)
print("Test set: ", accuracy_lr_test)
confusion_matrix(y_test, y_pred_lr_test)
dec_tree_classifier = DecisionTreeClassifier(
    criterion = 'gini', 
    max_features=8,
    random_state = 33)
dec_tree_classifier.fit(X_train, y_train)
# Predicting Cross Validation Score
cv_dt = cross_val_score(estimator = dec_tree_classifier, X = X_train, y = y_train.ravel(), cv = 10)
print("CV: ", cv_dt.mean())

y_pred_dt_train = dec_tree_classifier.predict(X_train)
accuracy_dt_train = accuracy_score(y_train, y_pred_dt_train)
print("Training set: ", accuracy_dt_train)

y_pred_dt_test = dec_tree_classifier.predict(X_test)
accuracy_dt_test = accuracy_score(y_test, y_pred_dt_test)
print("Test set: ", accuracy_dt_test)
confusion_matrix(y_test, y_pred_dt_test)
rfc = RandomForestClassifier(
    criterion = 'entropy', 
    max_features = 6, 
    n_estimators = 600, 
    random_state=33)
rfc.fit(X_train, y_train)
# Predicting Cross Validation Score
cv_rf = cross_val_score(estimator = rfc, X = X_train, y = y_train, cv = 10)
print("CV: ", cv_rf.mean())

y_pred_rf_train = rfc.predict(X_train)
accuracy_rf_train = accuracy_score(y_train, y_pred_rf_train)
print("Training set: ", accuracy_rf_train)

y_pred_rf_test = rfc.predict(X_test)
accuracy_rf_test = accuracy_score(y_test, y_pred_rf_test)
print("Test set: ", accuracy_rf_test)

confusion_matrix(y_test, y_pred_rf_test)
svc = svm.SVC(kernel = 'rbf',gamma = 'scale').fit(X_train, y_train) 
svc.fit(X_train, y_train)
y_pred_svc_test = svc.predict(X_test)
print(classification_report(y_test, y_pred_svc_test))
# Predicting Cross Validation Score
cv_svc = cross_val_score(estimator = svc, X = X_train, y = y_train.ravel(), cv = 10)
print("CV: ", cv_dt.mean())

y_pred_svc_train = svc.predict(X_train)
accuracy_svc_train = accuracy_score(y_train, y_pred_svc_train)
print("Training set: ", accuracy_svc_train)

y_pred_svc_test = svc.predict(X_test)
accuracy_svc_test = accuracy_score(y_test, y_pred_svc_test)
print("Test set: ", accuracy_svc_test)
confusion_matrix(y_test, y_pred_svc_test)
#Finding best parameters for our SVC model
params = {
    'C':[0.01,0.1,0.3,0.8,0.9,1,1.2,1.3,1.4,1.5,10],
    'kernel':['linear', 'rbf', 'sigmoid'],
    'gamma':[0.01,0.1,0.3,0.8,0.9,1,1.2,1.3,1.4,1.5,10]
}
grid_svc = GridSearchCV(svc, param_grid = params, scoring = 'accuracy', cv = 10)
grid_svc.fit(X_train, y_train)
grid_svc.best_params_
svc_grid= svm.SVC(C=1.2,kernel = 'rbf',gamma = 0.9).fit(X_train, y_train) 
svc_grid.fit(X_train, y_train)
# Predicting Cross Validation Score
cv_svc_grid = cross_val_score(estimator = svc_grid, X = X_train, y = y_train.ravel(), cv = 10)
print("CV: ", cv_svc.mean())

y_pred_svc_grid_train = svc_grid.predict(X_train)
accuracy_svc_train = accuracy_score(y_train, y_pred_svc_grid_train)
print("Training set: ", accuracy_svc_train)

y_pred_svc_grid_test = svc_grid.predict(X_test)
accuracy_svc_test = accuracy_score(y_test, y_pred_svc_grid_test)
print("Test set: ", accuracy_svc_test)
confusion_matrix(y_test, y_pred_svc_grid_test)

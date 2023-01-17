import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as  plt # background package for seaborn

import seaborn as sns # visualisation package based on plt

import sklearn # machine learning package

import os

print(os.listdir("../input"))

credit_df = pd.read_csv("../input/german_credit_data.csv",index_col=0)

credit_df.head(10)
credit_df.info()
credit_df = credit_df.fillna(value="not available")

credit_df.info()
credit_df.describe()
credit_df.nunique()
credit_df.Sex = credit_df.Sex.map({ 'male' : 1, 'female' : 2})
credit_df.Housing = credit_df.Housing.map({ 'own' : 1, 'rent' : 2, 'free' : 3})
credit_df['Saving accounts'] = credit_df['Saving accounts'].map({ 'not available' : 0, 'little' : 1, 'moderate' : 2, 'quite rich': 3, 'rich': 4})
credit_df['Checking account'] = credit_df['Checking account'].map({ 'not available' : 0, 'little' : 1, 'moderate' : 2, 'quite rich': 3, 'rich': 4})
credit_df['Purpose'] = credit_df['Purpose'].map({ 'car':1, 'furniture/equipment':2, 'radio/TV':3, 'domestic appliances':4, 'repairs':5, 'education':6, 'business':7, 'vacation/others':8})
credit_df.head(10)
sns.countplot(credit_df['Risk'], label = "Count") 
ax = sns.pairplot(credit_df, hue = 'Risk')
# Create set of only independant variables by dropping Risk

X = credit_df.drop(['Risk'], axis=1)

X.head()
# Create a series of outcome variable only

y = credit_df['Risk']

y.head()
# split datasets into training and test subsets for both X and y using sklearn

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)
print(X_train.shape)

print(X_test.shape)
from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix

# train the model

svc_model = SVC()

svc_model.fit(X_train, y_train)
y_pred = svc_model.fit(X_train, y_train).predict(X_test)
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True)
print(classification_report(y_test, y_pred))

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
min_train = X_train.min()

min_train
range_train = (X_train - min_train).max()

range_train
X_train_scaled = (X_train - min_train)/range_train

X_train_scaled.head()
sns.scatterplot(x = X_train['Credit amount'], y = X_train['Duration'], hue = y_train)
min_test = X_test.min()

range_test = (X_test - min_test).max()

X_test_scaled = (X_test - min_test)/range_test
svc_model1 = SVC()

svc_model1.fit(X_train_scaled, y_train)
y_predict = svc_model1.predict(X_test_scaled)

cm1 = confusion_matrix(y_test, y_predict)



sns.heatmap(cm1,annot=True,fmt="d")
accuracy_score(y_test, y_predict)
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)
grid.fit(X_train_scaled,y_train)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test_scaled)
cm = confusion_matrix(y_test, grid_predictions)

sns.heatmap(cm, annot=True)
print(classification_report(y_test,grid_predictions))
accuracy_score(y_test, grid_predictions)
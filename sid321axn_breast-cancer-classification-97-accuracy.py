# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import seaborn as sns
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_cancer = pd.read_csv("../input/breast-cancer.csv")
df_cancer.head()
df_cancer.loc[:,'diagnosis'] = df_cancer.diagnosis.map({'M':1, 'B':0})
df_cancer.head()
df_cancer.tail()
df_cancer.shape
df_cancer.isnull().sum()
sns.pairplot(df_cancer, hue = 'diagnosis', vars = ['radius_mean', 'texture_mean', 'area_mean', 'perimeter_mean', 'smoothness_mean'] )
sns.countplot(df_cancer['diagnosis'], label = "Count") 
sns.scatterplot(x = 'area_mean', y = 'smoothness_mean', hue = 'diagnosis', data = df_cancer)
sns.lmplot('area_mean', 'smoothness_mean', hue ='diagnosis', data = df_cancer, fit_reg=False)
fig = sns.FacetGrid(df_cancer, hue="diagnosis",aspect=4)

# Next use map to plot all the possible kdeplots for the 'Age' column by the hue choice
fig.map(sns.kdeplot,'smoothness_mean',shade= True)

# Set the x max limit by the oldest passenger
oldest = df_cancer['smoothness_mean'].max()

#Since we know no one can be negative years old set the x lower limit at 0
fig.set(xlim=(0,oldest))

#Finally add a legend
fig.add_legend()
fig = sns.FacetGrid(df_cancer, hue="diagnosis",aspect=4)

# Next use map to plot all the possible kdeplots for the 'Age' column by the hue choice
fig.map(sns.kdeplot,'texture_mean',shade= True)

# Set the x max limit by the oldest passenger
oldest = df_cancer['texture_mean'].max()

#Since we know no one can be negative years old set the x lower limit at 0
fig.set(xlim=(0,oldest))

#Finally add a legend
fig.add_legend()
fig = sns.FacetGrid(df_cancer, hue="diagnosis",aspect=4)

# Next use map to plot all the possible kdeplots for the 'Age' column by the hue choice
fig.map(sns.kdeplot,'area_mean',shade= True)

# Set the x max limit by the oldest passenger
oldest = df_cancer['area_mean'].max()

#Since we know no one can be negative years old set the x lower limit at 0
fig.set(xlim=(0,oldest))

#Finally add a legend
fig.add_legend()
sns.factorplot('texture_mean','area_mean',hue='diagnosis',data=df_cancer)
sns.scatterplot('concavity_se', 'radius_mean', hue ='diagnosis', data = df_cancer)

sns.scatterplot('compactness_se', 'radius_mean', hue ='diagnosis', data = df_cancer)
plt.figure(figsize=(24,12)) 
sns.heatmap(df_cancer.corr(), annot=True) 
unwantedcolumnlist=["diagnosis","Unnamed: 32","id"]
X = df_cancer.drop(unwantedcolumnlist,axis=1)
y = df_cancer['diagnosis']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=5)
from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()
svc_model.fit(X_train, y_train)
y_predict = svc_model.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
cm
sns.heatmap(cm, annot=True)
print(classification_report(y_test, y_predict))
min_train = X_train.min()
min_train
range_train = (X_train - min_train).max()
range_train
X_train_scaled = (X_train - min_train)/range_train
X_train_scaled.head()
sns.scatterplot(x = X_train['area_mean'], y = X_train['smoothness_mean'], hue = y_train)
sns.scatterplot(x = X_train_scaled['area_mean'], y = X_train_scaled['smoothness_mean'], hue = y_train)
min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test
from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()
svc_model.fit(X_train_scaled, y_train)
y_predict = svc_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_predict)

sns.heatmap(cm,annot=True,fmt="d")
print(classification_report(y_test,y_predict))
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)
grid.fit(X_train_scaled,y_train)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test_scaled)
cm = confusion_matrix(y_test, grid_predictions)
sns.heatmap(cm,annot=True,fmt="d")
print(classification_report(y_test,grid_predictions))
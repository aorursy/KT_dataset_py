# Importing Required Libraries
import numpy as np # linear algebra
import pandas as pd # data processing

import matplotlib.pyplot as plt # data viz
import seaborn as sns
%matplotlib inline
# Reading the Kyphosis data in the enviroment as df
df = pd.read_csv("../input/kyphosis-dataset/kyphosis.csv")
# Checking Data Head
df.head()
# Checking the info
df.info()
# 81 entries 
# small data set
# Count of Kyphosis
df['Kyphosis'].value_counts()
# Imblanced data set 
# may lead in wrong predictions
# Pair plot
sns.pairplot(df,hue = "Kyphosis")
from sklearn.model_selection import train_test_split
x = df.drop('Kyphosis',axis = 1)
y = df['Kyphosis']
x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                test_size=0.3)
# Importing Decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
dtree = DecisionTreeClassifier()
# Fitting decision tree on training data
dt = dtree.fit(x_train,y_train)
# Predicting on the Test Data set
predtree = dtree.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix
# Printing the confusion matrix
print(confusion_matrix(predtree,y_test))
print('\n')
print(classification_report(predtree,y_test))
# Plotting tree diagram of our decision tree.
from sklearn import tree
plt.figure(figsize=(20,20))
tree.plot_tree(dt)
# Importing Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
# Fitting the Model on training data 
rfc.fit(x_train,y_train)
# Precdicting of Random Forest on test data
predrand = rfc.predict(x_test)
# Confusion matrix for Random Forest
print(confusion_matrix(y_test,predrand))
print('\n')
print(classification_report(y_test,predrand))
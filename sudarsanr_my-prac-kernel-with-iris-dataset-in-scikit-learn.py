# My machine learning project with Iris dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# Importing Iris dataset from scikit learn library
from sklearn import datasets
iris = datasets.load_iris()
# Viewing data in a DataFrame
iris_data = iris.data
iris_data = pd.DataFrame(iris_data,columns=iris.feature_names)
iris_data['class'] = iris.target
iris_data.head()

# View target names of each flower
iris.target_names
# Understanding the data
print(iris_data.shape)
iris_data.describe()
# As this is a predefined dataset it contains equal (50) number of samples in each variety
# Analyse the data visually
# Box plot is a percentile-based graph, which divides the data into four quartiles of 25% each. This method is used in statistical analysis to understand various measures such as mean, median and deviation.
import seaborn as sns
sns.boxplot(data=iris_data,width=0.5,fliersize=5)
sns.set(rc={'figure.figsize':(8,15)})
# Apply algorithms
# First task is to split the data into training and test data
# train_test_split divides the data into 70:30 ratio
from sklearn.model_selection import train_test_split
X = iris_data.values[:,0:4]
Y = iris_data.values[:,4]
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=42)
# Some commonly used classification algorithms will be applied and tested here
# K-Nearest Neighbor (KNN).  Here we use KNN with number of neighbors 5
model = KNeighborsClassifier()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print(accuracy_score(y_test,predictions))


 # Support Vector Machine model works on the principle of Radial Basis function with default parameters. We will be using the RBF kernel to check the accuracy.
model = SVC()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print(accuracy_score(y_test,predictions))
# Randomforest is one of the highly accurate nonlinear algorithm, which works on the principle of Decision Tree Classification. 
model = RandomForestClassifier(n_estimators=2)
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print(accuracy_score(y_test,predictions))
# Logistic regression works on two schemes, first, if it is a binary classification problem, it works as one vs the rest, and if it is a multi class classification problem it works as one vs many.
model = LogisticRegression()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print(accuracy_score(y_test,predictions))

# Choose a model and Tune the parameters
# From the above models, we saw that randomforest gives us the best accuracy of 97.59%. Let us tune the parameter to get a 100% accuracy. Let us set the number of trees to be 500 to check if our model is performing well. (In the example.  Not here)
model = RandomForestClassifier(n_estimators=500)
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print(accuracy_score(y_test,predictions))


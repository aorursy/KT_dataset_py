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
# Importing the necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px 
import plotly.graph_objs as go
from plotly.offline import iplot

df = pd.read_csv("/kaggle/input/car-evaluation-data-set/car_evaluation.csv")
df.head()
# Total number of the columns in the Dataset
df.columns
df.shape
# Getting more information about the dataset
df.info()
# Describing the Dataset
df.describe()
# Checking whether there is any null values in the Dataset
df.isnull().sum()
# Since with the given name we are unable to judge what the data is indicating 
# Lets rename the columns name to understand the dataset more easily

df.columns = ['Price', 'Maintenance Cost', 'Number of Doors', 'Capacity', 'Size of Luggage Boot', 'safety', 'Decision']
# Lets read the data one more times to see how the data looks now
df.head()
# Visualizing the price Dataset

labels = df['Price'].value_counts().index
values = df['Price'].value_counts().values

colors = df['Price']

fig = go.Figure(data = [go.Pie(labels=labels, values=values, textinfo="label+percent",
                              insidetextorientation = "radial", marker=dict(colors=colors))])

fig.show()
# Visualizing the maintenance cost


labels = df['Maintenance Cost'].value_counts().index
values = df['Maintenance Cost'].value_counts().values

colors = df['Maintenance Cost']

fig = go.Figure(data = [go.Pie(labels=labels, values=values, textinfo="label+percent",
                              insidetextorientation = "radial", marker=dict(colors=colors))])

fig.show()
# Visualizing the distribution of number of Doors


labels = df['Number of Doors'].value_counts().index
values = df['Number of Doors'].value_counts().values

colors = df['Number of Doors']

fig = go.Figure(data = [go.Pie(labels=labels, values=values, textinfo="label+percent",
                              insidetextorientation = "radial", marker=dict(colors=colors))])

fig.show()
# Visualizing the distribution of number of Persons who can accomodate in the Car


labels = df['Capacity'].value_counts().index
values = df['Capacity'].value_counts().values

colors = df['Capacity']

fig = go.Figure(data = [go.Pie(labels=labels, values=values, textinfo="label+percent",
                              insidetextorientation = "radial", marker=dict(colors=colors))])

fig.show()
# Visualizing the dataset for Size of Luggage Boot


labels = df['Size of Luggage Boot'].value_counts().index
values = df['Size of Luggage Boot'].value_counts().values

colors = df['Size of Luggage Boot']

fig = go.Figure(data = [go.Pie(labels=labels, values=values, textinfo="label+percent",
                              insidetextorientation = "radial", marker=dict(colors=colors))])

fig.show()
# Visualizing the dataset for Size of Luggage Boot


labels = df['safety'].value_counts().index
values = df['safety'].value_counts().values

colors = df['safety']

fig = go.Figure(data = [go.Pie(labels=labels, values=values, textinfo="label+percent",
                              insidetextorientation = "radial", marker=dict(colors=colors))])

fig.show()
X = df.drop(['Decision'], axis = 1)
y = df['Decision']
# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print("The size of training input is", X_train.shape)
print("The size of training output is", y_train.shape)
print("The size of testing input is", X_test.shape)
print("The size of testing output is", y_test.shape)
# Importing the category Encoders

import category_encoders as ce
# Encoding the variables with ordinal encoding

encoder = ce.OrdinalEncoder(cols=['Price', 'Maintenance Cost', 'Number of Doors', 'Capacity', 'Size of Luggage Boot', 'safety'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)

# Now lets see how the Data looks like after doing the one-hot encoding

X_train.head()
X_test.head()
y_train.head()
y_test.head()
# Importing DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0)

# Fitting/Training the Model
clf.fit(X_train, y_train)
# Predicting Test set results using the Criterion Gini Index
y_pred = clf.predict(X_test)
# finding the training and testing accuracy
print("Training Accuracy: ",clf.score(X_train, y_train))
print("Testing Accuracy: ", clf.score(X_test, y_test))
from sklearn.metrics import confusion_matrix

# printing the confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# Printing the scores on training and test set

print('Training set Score: {:.4f}'.format(clf.score(X_train, y_train)))

print('Test set Score: {:.4f}'.format(clf.score(X_test, y_test)))
plt.figure(figsize=(12,8))

from sklearn import tree

tree.plot_tree(clf.fit(X_train, y_train))
import graphviz
dot_data = tree.export_graphviz(clf, out_file=None, feature_names = X_train.columns,
                               class_names = y_train, filled = True, rounded = True, 
                               special_characters = True)

graph = graphviz.Source(dot_data)

graph
clf2 = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)

# Fitting the model
clf2.fit(X_train, y_train)
y_pred = clf2.predict(X_test)
# finding the training and testing accuracy
print("Training Accuracy: ",clf2.score(X_train, y_train))
print("Testing Accuracy: ", clf2.score(X_test, y_test))
from sklearn.metrics import confusion_matrix

# printing the confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# Printing the scores on training and test set

print('Training set Score: {:.4f}'.format(clf2.score(X_train, y_train)))

print('Test set Score: {:.4f}'.format(clf2.score(X_test, y_test)))
plt.figure(figsize=(12,8))

from sklearn import tree

tree.plot_tree(clf2.fit(X_train, y_train))
import graphviz 
dot_data = tree.export_graphviz(clf2, out_file=None, 
                              feature_names=X_train.columns,  
                              class_names=y_train,  
                              filled=True, rounded=True,  
                              special_characters=True)

graph = graphviz.Source(dot_data) 

graph 
# Standardization

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Importing the required Libraries

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Defining model

model = LogisticRegression()

model.fit(X_train, y_train)

# Predicting the values for x-test
y_pred = model.predict(X_test)
# finding the training and testing accuracy
print("Training Accuracy: ",model.score(X_train, y_train))
print("Testing Accuracy: ", model.score(X_test, y_test))
# printing the confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# importing the Libraris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Creating a model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predicting the value for X_test
y_pred = model.predict(X_test)


# finding the training and testing accuracy
print("Training Accuracy: ",model.score(X_train, y_train))
print("Testing Accuracy: ", model.score(X_test, y_test))
# printing the confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.neighbors import KNeighborsClassifier

# creating a model
model = KNeighborsClassifier(n_neighbors = 5)

# feeding the training data into the model
model.fit(X_train, y_train)

# predicting the values for x-test
y_pred = model.predict(X_test)


# finding the training and testing accuracy
print("Training Accuracy: ",model.score(X_train, y_train))
print("Testing Accuracy: ", model.score(X_test, y_test))
# printing the confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

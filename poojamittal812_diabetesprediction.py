# DECISION TREE CLASSIFICATION 
# import the library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
 
# Read the csv file
df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.head()
# To check the observation and faeture
df.shape
# Here we separate the data into two parts one is our input and other is output where X is input and y is our output.
X = df.iloc[:,0:8].values
y = df.iloc[:,8].values
print(X)
print(y)
# Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

# Check the shape of our training and testing dataset
print(X_train.shape)
print(X_test.shape)
# Import Decision Tree Classifier for classification problem
from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier()
# Fit the data
DTC = DTC.fit(X_train,y_train)
# for prediction
y_pred = DTC.predict(X_test)

# Check accuracy
from sklearn.metrics import accuracy_score
print('Accuracy', accuracy_score(y_test,y_pred))

## Visualize the decision tree
from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image
import pydot
plt.figure(figsize=(10,10))
dot_data = StringIO()
export_graphviz(DTC, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=['Pregnancies','Glucose','BloodPressure', 'SkinThickness','Insulin',	'BMI','DiabetesPedigreeFunction','Age'], class_names=['0','1'])
graph = pydot.graph_from_dot_data(dot_data.getvalue())
Image(graph[0].create_png())
# Optimize decision tree performance by using pruning # 
DTC = DecisionTreeClassifier(criterion= 'entropy', max_depth=3)
DTC = DTC.fit(X_train,y_train)
y_pred = DTC.predict(X_test)
# again check accuracy score
print('Accuracy',accuracy_score(y_test,y_pred))
# again draw decision tree
dot_data = StringIO()
export_graphviz(DTC, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=['Pregnancies','Glucose','BloodPressure', 'SkinThickness','Insulin',	'BMI','DiabetesPedigreeFunction','Age'],class_names=['0','1'])
graph = pydot.graph_from_dot_data(dot_data.getvalue())
Image(graph[0].create_png())


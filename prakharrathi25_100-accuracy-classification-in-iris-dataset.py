import pandas as pd

import numpy as np

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
data = pd.read_csv("../input/Iris.csv")

data.shape
data.head(10)
data.tail(10)
data.isnull().any().any()
# Dropping the ID column

data.drop('Id',inplace=True,axis=1)
data.columns

cols = list(data.columns)

cols

range = data["SepalLengthCm"].max() - data["SepalLengthCm"].min()

range
data["SepalLengthCm"] = (data["SepalLengthCm"] - data["SepalLengthCm"].min())/range

data.head()
data["SepalLengthCm"] = data["SepalLengthCm"] / data["SepalLengthCm"].max()
data.head()
data = pd.read_csv("../input/Iris.csv")

data.shape
data.head()
group_names = data['Species'].unique().tolist()

group_names
%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt
sns.scatterplot(x = 'SepalLengthCm', y = 'SepalWidthCm', data = data, hue = 'Species')

plt.title('Sepal Length vs Sepal Width')

plt.show()
data['SepalLengthCm'].corr(data['SepalWidthCm'])
sns.scatterplot(x = 'PetalLengthCm', y = 'PetalWidthCm', data = data ,hue ='Species')

plt.title('Petal Length vs Petal Width')

plt.show()
data['PetalLengthCm'].corr(data['PetalWidthCm'])
sns.scatterplot(x = 'PetalLengthCm', y = 'SepalLengthCm', data = data ,hue ='Species')

plt.title('Petal Length vs Sepal Length')

plt.show()
data['PetalLengthCm'].corr(data['SepalLengthCm'])
sns.scatterplot(x = 'PetalWidthCm', y = 'SepalWidthCm', data = data ,hue ='Species')

plt.title('Petal Length vs Sepal Length')

plt.show()
data['PetalWidthCm'].corr(data['SepalWidthCm'])
sns.boxplot(x = "Species", y = "PetalLengthCm", data = data)
no_id_data = data.copy()

no_id_data.drop("Id", axis = 1, inplace = True)

sns.heatmap(data = no_id_data.corr(), annot = True)

plt.show()
x_values = data['PetalLengthCm'].copy()

y_values = data['PetalWidthCm'].copy()
x_train, x_test, y_train1, y_test1 = train_test_split(x_values, y_values, test_size = 0.33, random_state = 3)
from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
species_dummy = pd.get_dummies(data["Species"])

species_dummy.head()
assigned_data = data.copy()
assigned_data = pd.concat([data, species_dummy], axis = 1)

assigned_data.head()
assigned_data.drop(["Id"], inplace = True, axis = 1)

assigned_data.head()
target = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

features = cols[0:4]

print(target)

print(features)
y = assigned_data[target].copy()

X = assigned_data[features].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 3)
print(X_train.describe())

X_train.head()
y_train.head(10)
iris_classifier = DecisionTreeClassifier(max_leaf_nodes = 4, random_state = 0)

iris_classifier.fit(X_train, y_train)
y_prediction = iris_classifier.predict(X_test)
y_prediction[0 : 10]
y_test[0:10]
accuracy_score(y_true = y_test, y_pred = y_prediction)
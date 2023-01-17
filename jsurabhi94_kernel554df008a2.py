# importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from pandas.tools.plotting import parallel_coordinates

% matplotlib inline

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
# importing the dataset

data = pd.read_csv('../input/Iris.csv', index_col = 'Id')

data.head()
# summarizing the dataset

data.describe()
data.info()
plt.figure(figsize = (12,7))

parallel_coordinates(data, 'Species', color = ('cyan', 'yellow', 'green'), axvlines = True)

plt.xlabel('Features')

plt.ylabel('cm')

plt.title('Visualization of Iris data according to the features')
sns.heatmap(data.corr(), annot = True)
plt.figure(figsize = (12,7))

sns.scatterplot(x= 'SepalLengthCm', y = 'SepalWidthCm', hue = 'Species', data = data, size = 'Species')
plt.figure(figsize = (12,7))

sns.scatterplot(x= 'PetalLengthCm', y = 'PetalWidthCm', hue = 'Species', data = data)
plt.figure(figsize = (15,10))



plt.subplot(2,2,1)

sns.distplot(data.SepalWidthCm, kde = False, label = 'Sepal Width', color = 'red')

plt.legend()



plt.subplot(2,2,2)

sns.distplot(data.PetalWidthCm, kde = False, label = 'Petalal Width', color = 'blue')

plt.legend()



plt.subplot(2,2,3)

sns.distplot(data.SepalLengthCm, kde = False, label = 'Sepal Length', color = 'yellow')

plt.legend()



plt.subplot(2,2,4)

sns.distplot(data.PetalLengthCm, kde = False, label = 'Petal Length', color = 'black')

plt.legend()

x = data.iloc[:, 0:4]

y = data.Species

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.35, random_state = 2)
model = DecisionTreeClassifier(max_leaf_nodes = 5, random_state = 1)

model.fit(x_train, y_train)

y_val = model.predict(x_test)

    

print(accuracy_score(y_test, y_val))
model = KNeighborsClassifier(n_neighbors = 5, random_state = 1)

model.fit(x_train, y_train)

y_val = model.predict(x_test)

print(accuracy_score(y_test, y_val))
model = RandomForestClassifier(max_depth = 6, random_state = 1)

model.fit(x_train, y_train)

y_val = model.predict(x_test)

print(accuracy_score(y_test, y_val))
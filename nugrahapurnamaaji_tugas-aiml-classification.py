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
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

from matplotlib import pyplot as plt
import seaborn as sns

import graphviz

%matplotlib inline

df = pd.read_csv("/kaggle/input/flags-of-world/flag.csv", names = ['name', 'landmass', 'zone', 'area', 'population', 'language', 'religion', 'bars', 'stripes', 'colours', 'red', 'green', 'blue', 'gold', 'white', 'black', 'orange', 'mainhue', 'circles', 'crosses', 'saltires', 'quarters', 'sunstars', 'crescent', 'triangle', 'icon', 'animate', 'text', 'topleft', 'botright'])

df.head()
df.info()
df["mainhue"].value_counts()
table = df[['red', 'green', 'blue', 'gold', 'white', 'black', 'orange', 'mainhue']]
table.head()
feature = df.columns[10:17]
feature = list(feature)
feature
# Men-klasifikasi warna bendera
X = df[feature]
y = df['mainhue']
# Split to train and test our data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
dot_data = tree.export_graphviz(clf, out_file=None, feature_names = feature, class_names = y, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph
# How to predict our model
pred = clf.predict(X_test)
# Confusion Matrix
print(confusion_matrix(y_test, pred))
# Classification Report
print(classification_report(y_test, pred))
print("Accuracy = ", accuracy_score(y_test, pred))
accuracy_rate = []
for i in range(1, 40):
  tree = DecisionTreeClassifier(max_depth = i)
  tree.fit(X_train, y_train)
  accuracy_rate.append(tree.score(X_test, y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1, 40),accuracy_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Accuracy Rate vs. Tree Depth')
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy Rate')
from sklearn.neighbors import KNeighborsClassifier
# Mencari K terbaik
k_range = range(1, 41)
plt.figure(figsize=(10,6))
accuracy = {}
accuracy_rate = []
for k in k_range:
    knn = KNeighborsClassifier(k)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    accuracy[k] = accuracy_score(y_test, pred)
    # accuracy_rate.append(score.mean())
    accuracy_rate.append(accuracy_score(y_test, pred))

plt.plot(k_range,accuracy_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Accuracy Rate vs. K')
plt.xlabel('K')
plt.ylabel('Accuracy Rate')
print("Maximum accuracy:-",max(accuracy_rate),"at K =",accuracy_rate.index(max(accuracy_rate)))
plt.figure(figsize=(10,6))
error_rate = []
for k in k_range:
    clf = KNeighborsClassifier(k)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    error_rate.append(np.mean(pred != y_test))

plt.plot(k_range,error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K')
plt.xlabel('K')
plt.ylabel('Error Rate')
print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate)))
# Input K

n_neighbors = 33
knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
precision_recall_fscore_support(y_test, pred, average='macro')
precision_recall_fscore_support(y_test, pred, average='micro')
precision_recall_fscore_support(y_test, pred, average='weighted')
precision_recall_fscore_support(y_test, pred, average=None)
# d = {'red': 0, 'green': 1, 'blue': 2, 'gold': 3, 'white': 4, 'black': 5, 'orange': 6}
# y = y.map(d).astype('category')
clf = KNeighborsClassifier(n_neighbors)
clf.fit(X, y)

# Membuat prediksi dari input baru
# Input dalam angka bool 0 = No; 1 = Yes
red = int(input("Red: "))
green = int(input("Green: "))
blue = int(input("Blue: "))
gold = int(input("Gold: "))
white = int(input("White: "))
black = int(input("Black: "))
orange = int(input("Orange: "))
pred = clf.predict([[red, green, blue, gold, white, black, orange]])
print('Prediction Color: '),

if pred == 'red':
    print('Red')
elif pred == 'green':
    print('Green')
elif pred == 'blue': 
    print('Blue')
elif pred == 'gold':
    print('Gold')
elif pred == 'white':
    print('White')
elif pred == 'black':
    print('Black')
elif pred == 'orange':
    print('Orange')
else:
    print('Brown')
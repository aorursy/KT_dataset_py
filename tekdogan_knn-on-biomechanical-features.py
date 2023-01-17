import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))
data = pd.read_csv("../input/column_2C_weka.csv")
data.info()
data.head()
data['class'].value_counts()
data['class'] = [1 if each == 'Abnormal' else 0 for each in data['class']]
data.head()
f, ax = plt.subplots(figsize = (10,10))
sns.heatmap(data.corr(), annot=True, linewidths=.4, fmt= '.2f',ax=ax)
plt.show()
from pandas.plotting import scatter_matrix

scatter_matrix(data, alpha = 0.8, figsize = (15,15))
plt.show()
y = data['class']
x = data.drop(['class'], axis = 1)
x = (x - np.min(x))/(np.max(x) - np.min(x))
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

knn.fit(x_train, y_train)
score_list = []

for each in range (1,15):
    knn_o = KNeighborsClassifier(n_neighbors = each)
    knn_o.fit(x_train, y_train)
    score_list.append(knn_o.score(x_test, y_test))

plt.figure(figsize = (10,10))
plt.plot(range(1,15), score_list)
plt.xlabel('k values')
plt.ylabel('accuracy')
plt.show()
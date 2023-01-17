import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from pandas.tools.plotting import scatter_matrix

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.metrics import mean_squared_error
zoo = pd.read_csv('../input/zoo.csv', index_col='animal_name')
zoo.shape
zoo.head()
zoo.describe()
zoo.info()
corr = zoo.corr()



sns.heatmap(corr, square=True, linewidths=.3,cmap="RdBu_r")

plt.show()
corr_filt = corr[corr != 1][abs(corr)> 0.7].dropna(how='all', axis=1).dropna(how='all', axis=0)

print(corr_filt)
print("Correlation of Milk and Hair: %1.3f" %corr.loc['milk', 'hair'])

print("-------------------------")

print("Correlation of Milk and Toothed: %1.3f" %corr.loc['milk', 'toothed'])

print("-------------------------")

print("Correlation of Tail and Backbone: %1.3f" %corr.loc['tail', 'backbone'])

print("-------------------------")

print("Correlation of Milk and Eggs: %1.3f" %corr.loc['milk', 'eggs'])

print("-------------------------")

print("Correlation of Hair and Eggs: %1.3f" %corr.loc['hair', 'eggs'])

print("-------------------------")

print("Correlation of Hair and Eggs: %1.3f" %corr.loc['feathers', 'eggs'])

print("-------------------------")
plt.hist(zoo.class_type, bins=7)

plt.show()
zoo.class_type.value_counts()
zoo_sel = zoo.drop(['eggs', 'hair'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(zoo_sel.drop('class_type', axis=1), zoo_sel['class_type'], test_size=0.33, random_state=42)

mult = LogisticRegression()
mult.fit(X_train, y_train)
mult_pred = mult.predict(X_test)
conf_matrix_mult = confusion_matrix(y_test,mult_pred)

print(conf_matrix_mult)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7, leaf_size=50)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
conf_matrix_knn = confusion_matrix(y_test,knn_pred)

print(conf_matrix_knn)
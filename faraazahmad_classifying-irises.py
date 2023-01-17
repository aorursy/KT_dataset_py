# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
# loading dataset
iris_df = datasets.load_iris()
# list available features
print(iris_df.feature_names)

# list target_names
print(iris_df.target_names)
# Dataset Slicing
x_axis = iris_df.data[:, 0]     # sepal length
y_axis = iris_df.data[:, 2]     # sepal width
# Plotting with original clusters
plt.scatter(x_axis, y_axis, c=iris_df.target)
plt.show()
# Declaring model
model = KMeans(n_clusters=3)

# Fitting model
model.fit(iris_df.data)


# prediction on the entire data
all_predictions = model.predict(iris_df.data)
# plot the predictions of the model
plt.scatter(x_axis, y_axis, c=all_predictions)
plt.show()
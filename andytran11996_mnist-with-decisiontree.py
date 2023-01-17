import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler 
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split # Aufteilung in Training und Test
from sklearn.tree import DecisionTreeClassifier
mnist = fetch_mldata('MNIST original') # images of hand written digits
train_x, test_x, train_y, test_y = train_test_split( mnist.data, mnist.target, test_size=1/7.0, random_state=0)

scaler = StandardScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x)
test_x = scaler.transform(test_x) 
test_x_copy = test_x
import matplotlib.pyplot as plt
first_array=test_x_copy[0].reshape(28,28)
plt.imshow(first_array)
plt.show()
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
pca.fit(train_x)
train_x = pca.transform(train_x)
test_x = pca.transform(test_x)
model = DecisionTreeClassifier(criterion="gini", max_depth=32)
model.fit(train_x, train_y)
score = model.score(test_x, test_y)
print(score)
model2 = DecisionTreeClassifier(criterion="entropy", max_depth=32)
model2.fit(train_x, train_y)
score = model2.score(test_x, test_y)
print(score)
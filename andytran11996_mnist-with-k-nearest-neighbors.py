import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler 
# Load the data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train_x, train_y = train.values[:,1:], train.values[:,0]
test_x = test.values
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
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10, n_jobs = -1)
knn.fit(train_x, train_y)
results = knn.predict(test_x)
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),pd.Series(results, name='Label')],axis = 1)
submission.to_csv("cnn_mnist_datagen.csv",index=False)

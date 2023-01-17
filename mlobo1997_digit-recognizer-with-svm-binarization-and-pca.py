import pandas as pd
import numpy as np
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from scipy.sparse import lil_matrix
from sklearn import svm
from sklearn.decomposition import PCA
import sklearn.discriminant_analysis
data = pd.read_csv("../input/train.csv")
data.shape
i = 6
image = np.array(data.iloc[i,1:])
image = image.reshape([28, 28])
plt.imshow(image, cmap='gray')
plt.title(data.iloc[i,0])
train_n = 5000
train_labels = np.array(data.iloc[:train_n,0])
train = lil_matrix(np.array(data.iloc[:train_n, 1:]), dtype = 'int32')
train_labels.shape, train.shape
test_n = 10000
test_labels = np.array(data.iloc[train_n : train_n + test_n, 0])
test = lil_matrix(np.array(data.iloc[train_n : train_n + test_n, 1:]), dtype = 'int32')
test_labels.shape, test.shape
clf = svm.SVC(gamma='scale')
clf.fit(train, train_labels)
clf.score(test, test_labels)
data_simple = (np.array(data)[:,1:] >= 120).astype(int)
data_simple.shape
i = 3
image = data_simple[i,:]
image = image.reshape([28, 28])
plt.imshow(image, cmap='gray')
plt.title(data.iloc[i,0])
train_n = 32000
train_labels = np.array(data.iloc[:train_n,0])
train = data_simple[:train_n]
train_labels.shape, train.shape
test_n = 10000
test_labels = np.array(data.iloc[train_n : train_n + test_n, 0])
test = data_simple[train_n : train_n + test_n]
test_labels.shape, test.shape
clf2 = svm.SVC(gamma='scale')
clf2.fit(train, train_labels)
clf2.score(test, test_labels)
pca = PCA(0.65)
pca.fit(train)
pca.n_components_
train_pca = pca.transform(train)
test_pca = pca.transform(test)
train_pca.shape, test_pca.shape
clf3 = svm.SVC(gamma='scale')
clf3.fit(train_pca, train_labels)
clf3.score(test_pca, test_labels)
train = data_simple
train_labels = np.array(data.iloc[:,0])
test = (pd.read_csv('../input/test.csv') >= 120).astype(int)
train.shape, train_labels.shape, test.shape
pca = PCA(0.65)
pca.fit(train)
pca.n_components_
train_pca = pca.transform(train)
test_pca = pca.transform(test)
train_pca.shape, test_pca.shape
clf4 = svm.SVC(gamma='scale')
clf4.fit(train_pca, train_labels)
pred = clf4.predict(test_pca)
pred.shape
r = np.array([range(1,28001), pred], dtype = int).transpose()
r = pd.DataFrame(r)
r.columns = ["ImageId", "Label"]
r
r.to_csv("submit.csv", index = False)
help(r.to_csv)

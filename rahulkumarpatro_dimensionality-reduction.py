# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train=pd.read_csv("/kaggle/input/mnist-in-csv/mnist_train.csv")
df_train
df_test=pd.read_csv("/kaggle/input/mnist-in-csv/mnist_test.csv")
df_test
df_train.shape
df_train.info()
df_train.describe()
df_train.isnull().sum()
df_train.columns
df_test.columns
X_train=df_train.drop('label',axis=1)
Y_train=df_train['label']
X_test=df_test.drop('label',axis=1)
Y_test=df_test['label']
# X_train
plt.imshow(X_train.iloc[9000].to_numpy().reshape(28,28))
Y_train[9000]
#we will keep track of the time taken to perform each transformation and training.
import time
from sklearn.decomposition import PCA
pca=PCA(n_components=0.95)
start = time.time()

X_red= pca.fit_transform(X_train)
end = time.time()

end - start
X_train.shape
X_red.shape
pca.n_components_
plt.imshow(X_train.iloc[9000].to_numpy().reshape(28,28))
plt.imshow(X_red[1].reshape(7,22))
X_return=pca.inverse_transform(X_red)
X_return.shape
plt.imshow(X_return[9000].reshape(28,28))
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.imshow(X_train.iloc[9000].to_numpy().reshape(28,28))
ax2.imshow(X_return[9000].reshape(28, 28))
fig.suptitle('Compression and Decompression')
ax1.axis('off')
ax2.axis('off')
plt.show()
# First trying for Logistic Regression

from sklearn.linear_model import LogisticRegression

log_clf = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs', random_state = 42)
t_start = time.time()
log_clf.fit(X_train, Y_train)
t_end= time.time()
t_end-t_start
log_clf.score(X_test, Y_test)
from sklearn.linear_model import LogisticRegression

log_clf = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs', random_state = 42)
t_start = time.time()
log_clf.fit(X_red, Y_train)
t_end= time.time()
t_end-t_start
X_red_test = pca.transform(X_test)
log_clf.score(X_red_test, Y_test)
X_red_test.shape
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, random_state = 42)
t_start = time.time()
rfc.fit(X_train,Y_train)
t_end = time.time()
t_end-t_start
rfc.score(X_test, Y_test)
rfc = RandomForestClassifier(n_estimators=100, random_state = 42)
t_start = time.time()
rfc.fit(X_red,Y_train)
t_end = time.time()
t_end-t_start
rfc.score(X_red_test, Y_test)
X_train['label'] = Y_train
X = X_train.sample(n=10000, random_state=42)

Y = X['label']
X = X.drop('label', axis = 1)
X.shape
Y.shape
#We will use TSNE to reduce the datset down to 2 Dimensions and then plot it using Matplotlib
from sklearn.manifold import TSNE
tsne = TSNE(n_components = 2, random_state = 42)
t_start = time.time()
X_reduced = tsne.fit_transform(X)
t_end = time.time()
t_end-t_start
plt.figure(figsize=(12, 8))
plt.scatter(X_reduced[:,0], X_reduced[:,1], c = Y, cmap='jet')
plt.colorbar()
plt.axis('off')
plt.show()
from sklearn.pipeline import Pipeline
# Using Pipelines..
pca_tsne = Pipeline([
    ('pca', PCA(n_components=0.95, random_state=42)),
    ('tsne', TSNE(n_components=2, random_state=42)),
])
t_start = time.time()
X_new = pca_tsne.fit_transform(X)
t_end = time.time()
print(t_end-t_start)

plt.figure(figsize=(12, 8))
plt.scatter(X_new[:,0], X_new[:,1], c = Y, cmap='jet')
plt.colorbar()
plt.axis('off')
plt.show()
from sklearn.manifold import LocallyLinearEmbedding
t_start = time.time()
X_lle = LocallyLinearEmbedding(n_components=2, random_state=42).fit_transform(X)
t_end = time.time()
print(t_end-t_start)

plt.figure(figsize=(12, 8))
plt.scatter(X_lle[:,0], X_lle[:,1], c = Y, cmap='jet')
plt.colorbar()
plt.axis('off')
plt.show()
pca_lle = Pipeline([
    ("pca", PCA(n_components=0.95, random_state=42)),
    ("lle", LocallyLinearEmbedding(n_components=2, random_state=42)),
])
t_start = time.time()
X_new1= pca_lle.fit_transform(X)
t_end = time.time()
print(t_end-t_start)

plt.figure(figsize=(12, 8))
plt.scatter(X_new1[:,0], X_new1[:,1], c = Y, cmap='jet')
plt.colorbar()
plt.axis('off')
plt.show()
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

t_start = time.time()
X_lda = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, Y)
t_end = time.time()
print(t_end-t_start)

plt.figure(figsize=(12, 8))
plt.scatter(X_lda[:,0], X_lda[:,1], c = Y, cmap='jet')
plt.colorbar()
plt.axis('off')
plt.show()

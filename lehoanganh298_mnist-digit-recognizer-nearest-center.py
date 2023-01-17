import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

train_data = pd.read_csv('../input/train.csv');
test_data = pd.read_csv('../input/test.csv');
test_X=test_data.values
train_y=train_data['label'].values
train_X=train_data.drop('label',axis = 1).values
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


# This function visualizes filters in matrix A. Each column of A is a
# filter. We will reshape each column into a square image and visualizes
# on each cell of the visualization panel.
# All other parameters are optional, usually you do not need to worry
# about it.
# opt_normalize: whether we need to normalize the filter so that all of
# them can have similar contrast. Default value is true.
# opt_graycolor: whether we use gray as the heat map. Default is true.
# opt_colmajor: you can switch convention to row major for A. In that
# case, each row of A is a filter. Default value is false.
# source: https://github.com/tsaith/ufldl_tutorial

def display_network(A, m = -1, n = -1):
    opt_normalize = True
    opt_graycolor = True

    # Rescale
    A = A - np.average(A)

    # Compute rows & cols
    (row, col) = A.shape
    sz = int(np.ceil(np.sqrt(row)))
    buf = 1
    if m < 0 or n < 0:
        n = np.ceil(np.sqrt(col))
        m = np.ceil(col / n)
        

    image = np.ones(shape=(buf + m * (sz + buf), buf + n * (sz + buf)))

    if not opt_graycolor:
        image *= 0.1

    k = 0

    for i in range(int(m)):
        for j in range(int(n)):
            if k >= col:
                continue

            clim = np.max(np.abs(A[:, k]))

            if opt_normalize:
                image[buf + i * (sz + buf):buf + i * (sz + buf) + sz, buf + j * (sz + buf):buf + j * (sz + buf) + sz] = \
                    A[:, k].reshape(sz, sz) / clim
            else:
                image[buf + i * (sz + buf):buf + i * (sz + buf) + sz, buf + j * (sz + buf):buf + j * (sz + buf) + sz] = \
                    A[:, k].reshape(sz, sz) / np.max(np.abs(A))
            k += 1

    return image
K=10
rows, columns = train_X.shape
centers = np.zeros((K,columns),dtype='int')
n_each_label = np.zeros((K,1),dtype='int')

for i in range(rows):
    centers[train_y[i]] += train_X[i]
    n_each_label[train_y[i]]+=1
    
for i in range(K):
    centers[i]=centers[i]/n_each_label[i]
    
centers
a=display_network(centers.T,K,1)
im=plt.imshow(a)
n_tests = test_data.shape[0]

from numpy.linalg import norm

test_y=np.zeros((n_tests,1),dtype='int')
for i in range(n_tests):
    dists=norm(test_X[i]-centers,axis=1)
    test_y[i]=dists.argmin()

df=pd.DataFrame(test_y)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('result.csv',header=True)
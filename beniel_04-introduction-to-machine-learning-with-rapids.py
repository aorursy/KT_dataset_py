import os
import subprocess

try:
    import matplotlib
except ModuleNotFoundError:
    os.system('conda install -y matplotlib')
    import matplotlib

!nvidia-smi
!nvcc --version
import sys
!rsync -ah --progress ../input/rapids/rapids.0.14.0 /opt/conda/envs/rapids.tar.gz
!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null
sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 
!rsync -ah --progress /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


%matplotlib inline
import numpy as np; print('NumPy Version:', np.__version__)


# create the relationship: y = 2.0 * x + 1.0

n_rows = 43000
w = 2.0
x = np.random.normal(loc=0, scale=1, size=(n_rows,))
b = 1.0
y = w * x + b

# add a bit of noise
noise = np.random.normal(loc=0, scale=2, size=(n_rows,))
y_noisy = y + noise
plt.scatter(x, y_noisy, label='empirical data points')
plt.plot(x, y, color='black', label='true relationship')
plt.legend()
import sklearn; print('Scikit-Learn Version:', sklearn.__version__)
from sklearn.linear_model import LinearRegression


# instantiate and fit model
linear_regression = LinearRegression()
%%time

linear_regression.fit(np.expand_dims(x, 1), y)
# create new data and perform inference
inputs = np.linspace(start=-5, stop=5, num=1000)
outputs = linear_regression.predict(np.expand_dims(inputs, 1))
plt.scatter(x, y_noisy, label='empirical data points')
plt.plot(x, y, color='black', label='true relationship')
plt.plot(inputs, outputs, color='red', label='predicted relationship (cpu)')
plt.legend()
import cudf; print('cuDF Version:', cudf.__version__)


# create a cuDF DataFrame
df = cudf.DataFrame({'x': x, 'y': y_noisy})
print(df.head())
import cuml; print('cuML Version:', cuml.__version__)
from cuml.linear_model import LinearRegression as LinearRegression_GPU


# instantiate and fit model
linear_regression_gpu = LinearRegression_GPU()
%%time

linear_regression_gpu.fit(df['x'], df['y'])
# create new data and perform inference
new_data_df = cudf.DataFrame({'inputs': inputs})
outputs_gpu = linear_regression_gpu.predict(new_data_df[['inputs']])
plt.scatter(x, y_noisy, label='empirical data points')
plt.plot(x, y, color='black', label='true relationship')
plt.plot(inputs, outputs, color='red', label='predicted relationship (cpu)')
plt.plot(inputs, outputs_gpu.to_array(), color='green', label='predicted relationship (gpu)')
plt.legend()
from sklearn.datasets import make_moons


X, y = make_moons(n_samples=int(1e3), noise=0.05, random_state=0)
print(X.shape)
figure = plt.figure()
axis = figure.add_subplot(111)
axis.scatter(X[y == 0, 0], X[y == 0, 1], 
             edgecolor='black',
             c='lightblue', marker='o', s=40, label='cluster 1')

axis.scatter(X[y == 1, 0], X[y == 1, 1], 
             edgecolor='black',
             c='red', marker='s', s=40, label='cluster 2')
plt.legend()
plt.tight_layout()
plt.show()
X_df = cudf.DataFrame()
for column in range(X.shape[1]):
    X_df['feature_' + str(column)] = np.ascontiguousarray(X[:, column])

y_df = cudf.Series(y)
from cuml.neighbors import NearestNeighbors


knn = NearestNeighbors()
knn.fit(X_df)
k = 3

distances, indices = knn.kneighbors(X_df, n_neighbors=k)
distances
indices
predictions = []

for i in range(indices.shape[0]):
    row = indices.iloc[i, :]
    vote = sum(y_df[j] for j in row) / k
    predictions.append(1.0 * (vote > 0.5))

predictions = np.asarray(predictions).astype(np.float32)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))


ax1.scatter(X[y == 0, 0], X[y == 0, 1],
            edgecolor='black',
            c='lightblue', marker='o', s=40, label='cluster 1')
ax1.scatter(X[y == 1, 0], X[y == 1, 1],
            edgecolor='black',
            c='red', marker='s', s=40, label='cluster 2')
ax1.set_title('empirical data points')


ax2.scatter(X[predictions == 0, 0], X[predictions == 0, 1], c='lightblue',
            edgecolor='black',
            marker='o', s=40, label='cluster 1')
ax2.scatter(X[predictions == 1, 0], X[predictions == 1, 1], c='red',
            edgecolor='black',
            marker='s', s=40, label='cluster 2')
ax2.set_title('KNN predicted classes')

plt.legend()
plt.tight_layout()
plt.show()
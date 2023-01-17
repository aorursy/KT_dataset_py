import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
dataset = datasets.load_boston()
m = dataset['target'].shape[0]
n = dataset['data'].shape[1]

# Scale the data - this is important!
scaler = StandardScaler()
X_scaled = np.c_[np.ones([m,1]), scaler.fit_transform(dataset['data'])]
y_vals = dataset['target'].reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_vals, test_size=0.2, random_state=42)
tf.reset_default_graph()
tf.set_random_seed(42)
# yur code here
# yur code here
# yur code here
# yur code here
# yur code here
X_all = np.c_[np.ones([m,1]), dataset['data']]
y_all = dataset['target']
theta_norm = np.linalg.inv(X_all.T.dot(X_all)).dot(X_all.T).dot(y_all)
norm_error = np.sum(np.square(y_all - X_all.dot(theta_norm))) / y_all.shape[0]
print('Norm equation error:',norm_error)
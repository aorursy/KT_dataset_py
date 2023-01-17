import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
base_dir = '/kaggle/input/';

# load features, ignore header, ignore IDs
X_train = np.loadtxt(base_dir + 'train_features.csv', delimiter=',')[:,1:];
X_test = np.loadtxt(base_dir + 'test_features.csv', delimiter=',')[:,1:];
y_train = np.loadtxt(base_dir + 'train_labels.csv', dtype=np.uint8, delimiter=',', skiprows=1)[:,-1];
im_train = X_train[0,:].reshape((30,30,3), order='F')
im_test = X_test[0,:].reshape((30,30,3), order='F')

plt.figure(1)
plt.imshow(im_train/255)
plt.axis('off')

plt.figure(2)
plt.imshow(im_test/255)
plt.axis('off');
clf = KNeighborsClassifier(n_jobs=-1)
clf.fit(X_train, y_train)
df = pd.DataFrame(clf.predict(X_test), columns=['Label'])
df.index += 1 # upgrade to one-based indexing
df.to_csv('knn_submission.csv',index_label='ID',columns=['Label'])
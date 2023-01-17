import numpy as np

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor



# load data, handle row ids

X_train = np.loadtxt("/kaggle/input/roxie2/roxie_train_features.csv", delimiter=",")[:,1:]

X_test = np.loadtxt("/kaggle/input/roxie2/roxie_test_features.csv", delimiter=",")

ids_test = X_test[:,(0,)]

X_test = X_test[:,1:]

y_train = np.loadtxt("/kaggle/input/roxie2/roxie_train_values.csv", delimiter=",", ndmin=2)[:,(1,)]



mdl = KNeighborsRegressor()

mdl.fit(X_train, y_train)

y_pred = mdl.predict(X_test)
# export predictions

output = np.concatenate((ids_test, y_pred), axis=1)

np.savetxt("submission_knn.csv", output, delimiter=",", fmt='%i,%1.4f', header='ID,intensity')
X_full = np.loadtxt("/kaggle/input/roxie2/roxie_full_features.csv", delimiter=",")

y_pred = mdl.predict(X_full)

plt.figure(figsize=(10,15))

plt.imshow(y_pred.reshape((650,430,3))/255)
import numpy as np

import matplotlib.pyplot as plt

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
base_dir = '/kaggle/input/roxie/'



X_train = np.loadtxt(base_dir + "roxie_train_features.csv", delimiter=",")[:,1:]

X_test = np.loadtxt(base_dir + "roxie_test_features.csv", delimiter=",")

ids_test = X_test[:,(0,)]

X_test = X_test[:,1:]

y_train = np.loadtxt(base_dir + "roxie_train_values.csv", delimiter=",", ndmin=2)[:,(1,)]
from sklearn.tree import DecisionTreeRegressor



# train model

mdl = DecisionTreeRegressor()

mdl.fit(X_train, y_train)



# make predictions on test data

y_pred = mdl.predict(X_test)
output = np.concatenate((ids_test, y_pred[:,np.newaxis]), axis=1)

np.savetxt("submission_decision_tree.csv", output, delimiter=",", fmt='%1.4f', header='ID,intensity')
# this file contains all pixels (the union of the train and test sets)

X_full = np.loadtxt(base_dir + "roxie_full_features.csv", delimiter=",")



# make predictions for all pixels

y_pred = mdl.predict(X_full)



# show it as an image

plt.figure(figsize=(10,15))

plt.imshow(y_pred.reshape((650,430,3)));
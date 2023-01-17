#--------------------------------------------------------

import os

print(os.listdir("../input"))

#--------------------------------------------------------

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt 

#--------------------------------------------------------

from sklearn import datasets



from keras import Sequential

from keras.layers import Dense, Dropout

from keras.optimizers import Adam

from keras.utils import to_categorical



from sklearn.model_selection import train_test_split
n_pts = 1500

centers = [[-2,2],[-2,-2],[2,-2]]

#---------------------------------------------------------------------

# to find out something about ret type and arguments of the method

#print(datasets.make_blobs.__doc__)

#---------------------------------------------------------------------

X, y = datasets.make_blobs(n_samples=n_pts, random_state=123, centers=centers, cluster_std=1)

print(X)

print(y)
plt.scatter(X[y==0,0], X[y==0,1])

plt.scatter(X[y==1,0], X[y==1,1])

plt.scatter(X[y==2,0], X[y==2,1])
print(y)

y_cat = to_categorical(y, 3)

print(y_cat)
print(y)

y_cat = to_categorical(y, 3)

print(y_cat)





X, X_val,Y, y_val = train_test_split(X, y_cat, test_size=0.1, random_state=2)

print("Shape of X :", X.shape)

print("Shape of y :", y.shape)
model = Sequential()

model.add(Dense(units=3, input_shape=(2,), activation="relu"))

model.add(Dropout(0.3))

model.add(Dense(units=8, activation="relu"))

model.add(Dropout(0.25))

model.add(Dense(units=6, activation="relu"))

model.add(Dropout(0.3))

model.add(Dense(units=3, activation="softmax"))

model.compile(Adam(0.001), loss="categorical_crossentropy", metrics=["accuracy"])
h = model.fit(x=X, y=Y, verbose=1, batch_size=50, epochs=500, validation_data=(X_val,y_val))
print(h.history.keys())

plt.figure(figsize=(13,7))

plt.plot(h.history["acc"])

plt.plot(h.history["val_acc"])

plt.legend(["accuracy", "validation accuracy"])

plt.show()
def plot_multiclass_decision_boundary(X, y, model):

    x_span = np.linspace(min(X[:,0]) - 1, max(X[:,0]) + 1)

    y_span = np.linspace(min(X[:,1]) - 1, max(X[:,1]) + 1)

    xx, yy = np.meshgrid(x_span, y_span)

    grid = np.c_[xx.ravel(), yy.ravel()]

    pred_func = model.predict_classes(grid)

    z = pred_func.reshape(xx.shape)

    plt.contourf(xx, yy, z)
plot_multiclass_decision_boundary(X, Y, model)



point = np.array([[-2,2]])



plt.scatter(point[0][0],point[0][1], color="black")



ret = model.predict_classes(point)



if ret == 1:

    print("point belongs to the green area")

if ret == 0:

    print("point belongs to the magenta area")

if ret == 2:

    print("point belongs to the yellow area")
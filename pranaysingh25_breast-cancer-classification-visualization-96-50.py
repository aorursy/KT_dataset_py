#importing essential libraries

import numpy as np

import pandas as pd 

import matplotlib 

import matplotlib.pyplot as plt
#loading the dataset

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

#splitting our data into training and testing data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)
#checking the shape of the data

X_train.shape, X_test.shape, y_train.shape, y_test.shape
#the names of the feature used

feature_names = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean','smoothness_mean','compactness_mean',

                                           'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal dimension_mean', 

                                           'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se',

                                           'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 

                                           'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',

                                           'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal dimension_worst' ]
X_train = pd.DataFrame(X_train, columns = feature_names)
X_train.head(5)
#lets see if there are any missing values

X_train.info()
#checking the scale of the data

X_train.describe()
#lets see how our data looks like on the graph

X_train.hist(bins=50, figsize=(20,15))

plt.show()
#we need to scale the data

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

X_train = scale.fit_transform(X_train)

X_test = scale.transform(X_test)
X_train = pd.DataFrame(X_train, columns= feature_names)

X_test = pd.DataFrame(X_test, columns = feature_names)
#making the copy of our data before any transformation

X_train_c = X_train.copy()
#lets try first The PCA 

from sklearn.decomposition import PCA

pca = PCA(n_components = 2)

X_train2d = pca.fit_transform(X_train_c)
#about 63% of the variance has been preserved by only 2 features!

pca.explained_variance_ratio_
#lets now finally plot these components against the labels 

plt.figure(figsize=(12,8))

plt.scatter(X_train2d[:,0], X_train2d[:,1], c=y_train, cmap='jet')

plt.axis('off')

plt.colorbar()

plt.show()
#TSNE

from sklearn.manifold import TSNE



tsne= TSNE(n_components = 2, random_state = 42)

X_train_tsne= tsne.fit_transform(X_train_c)
#lets see how far tsne gets

plt.figure(figsize=(12,8))

plt.scatter(X_train_tsne[:,0], X_train_tsne[:,1], c=y_train, cmap='jet')

plt.axis('off')

plt.colorbar()

plt.show()
#starting with Stochastic Gradient Descent classifier

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42, early_stopping= True, n_iter_no_change=10)
#using cross validation 

from sklearn.model_selection import cross_val_score

cvs = cross_val_score(sgd_clf, X_train, y_train, cv=5)

print(cvs)

print(cvs.mean())

print(cvs.std())
#trying Decision Tree

from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier(random_state=42)

cvs2 = cross_val_score(dt_clf, X_train, y_train, cv=5)

print(cvs2)

print(cvs2.mean())

print(cvs2.std())
#lets try random forest with as many as 100 decition trees!

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=42, n_estimators=100)



cvs3 = cross_val_score(rfc, X_train, y_train, cv=5)

print(cvs3)

print(cvs3.mean())

print(cvs3.std())
#a shallow NN with 2 hidden layers

#defining a NN using Sequential API

from tensorflow import keras

model = keras.models.Sequential([

    keras.layers.Dense(300, activation = 'relu', input_shape=(30,)),

    keras.layers.Dense(100, activation='relu'),

    keras.layers.Dense(1, activation='sigmoid')

])
#compiling our model with same loss and accuracy metric used for SGD

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
#have a look how it looks

model.summary()

keras.utils.plot_model(model)
#let's train it

early_stopping_cb = keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=35, validation_split=0.1, callbacks= [early_stopping_cb])
#we have the 'losses' history.history, we can plot it after converting it into dataframe.

pd.DataFrame(history.history).plot(figsize=(8,5))

plt.grid(True)

plt.gca().set_ylim(0,1)

plt.show()
#now evaluating on the test data

model.evaluate(X_test, y_test)
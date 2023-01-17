import pandas as pd

import numpy as np

import plotly as py

import sklearn as skl



py.offline.init_notebook_mode(connected=True)

np.set_printoptions(linewidth=200)
data = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



response = data['label']

features = data.drop(columns=['label'], axis=1)
import sklearn.preprocessing

import sklearn.decomposition



features = skl.preprocessing.StandardScaler().fit_transform(features)

pca = skl.decomposition.PCA(.90).fit(features)



features = pca.transform(features)

test = pca.transform(test)
from sklearn.metrics import accuracy_score

import sklearn.model_selection
import sklearn.neighbors



knn_model = skl.neighbors.KNeighborsClassifier(n_neighbors=10).fit(features, response)

predictions = knn_model.predict(test)

knn_predicitons = pd.DataFrame({

    'ImageId':pd.Series(np.arange(1, 28001, 1)),

    'Label':predictions

})
import sklearn.svm



svm_model = skl.svm.SVC(kernel='poly', gamma='auto').fit(features, response)

predictions = svm_model.predict(test)

svm_predictions = pd.DataFrame({

    'ImageId':pd.Series(np.arange(1, 28001, 1)),

    'Label':predictions

})
import sklearn.tree



kf = skl.model_selection.KFold(n_splits=2)



cv_model = skl.model_selection.cross_val_score(skl.tree.DecisionTreeClassifier(), features, response, cv=kf)



dtree_model = skl.tree.DecisionTreeClassifier().fit(features, response)

predictions = dtree_model.predict(test)

dtree_predictions = pd.DataFrame({

    'ImageId':pd.Series(np.arange(1, 28001, 1)),

    'Label':predictions

})
import tensorflow as tf

import tensorflow.keras
neural_network = tf.keras.models.Sequential()

neural_network.add(tf.keras.layers.Flatten())

neural_network.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

neural_network.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

neural_network.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))



neural_network.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)
neural_network.fit(features, response.values, epochs=3)
probability_predictions = neural_network.predict(test)

predictions = []

for i, p in enumerate(probability_predictions):

    predictions.append(np.argmax(probability_predictions[i]))



neural_network_predicitons = pd.DataFrame({

    'ImageId':pd.Series(np.arange(1, 28001, 1)),

    'Label':predictions

})
dtree_predictions
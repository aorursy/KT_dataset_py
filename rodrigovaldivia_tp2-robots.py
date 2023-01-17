# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import mean_squared_error



from sklearn import tree

import matplotlib.pyplot as plt
data = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv');

print(data.head())
CATEGORIES = {

    'class': ['e', 'p'],

    'cap-shape': ['x', 'b', 'c', 'k', 's', 'f'], 

    'cap-surface': ['f', 'g', 'y', 's'], 

    'cap-color': ['n', 'b', 'c', 'g','r', 'p', 'u', 'e', 'w', 'y'], 

    'bruises': ['t', 'f'], 

    'odor': ['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'],

    'gill-attachment': ['a', 'd', 'f', 'n'], 

    'gill-spacing': ['c', 'w', 'd'], 

    'gill-size': ['b', 'n'],

    'gill-color': ['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y'],

    'stalk-shape': ['e', 't'], 

    'stalk-root': ['b', 'c', 'u', 'e', 'z', 'r', '?'], 

    'stalk-surface-above-ring': ['f', 'y', 'k', 's'],

    'stalk-surface-below-ring': ['f', 'y', 'k', 's'],

    'stalk-color-above-ring': ['b', 'n', 'c', 'g', 'o', 'p', 'e', 'w', 'y'],

    'stalk-color-below-ring': ['b', 'n', 'c', 'g', 'o', 'p', 'e', 'w', 'y'],

    'veil-type': ['p', 'u'], 

    'veil-color': ['n', 'o', 'w', 'y'], 

    'ring-number': ['o', 'n', 't'],

    'ring-type': ['c', 'e', 'f', 'l', 'n', 'p', 's', 'z'], 

    'spore-print-color': ['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y'], 

    'population': ['a', 'c', 'n', 's', 'v', 'y'], 

    'habitat': ['g', 'l', 'm', 'p', 'u', 'w', 'd']  

}
    

df = data

for feature, vocab in CATEGORIES.items():

    df[feature] = df[feature].apply(lambda x: vocab.index(x))

df.head()
X = df.iloc[:, 1:23].values

y = df.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=40)

X_train.shape
model = tf.keras.Sequential([

    tf.keras.layers.Flatten(input_shape=(22,)),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(2),

])



model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20)
y_predict = model.predict(X_test).argmax(axis=-1)

metrics = precision_recall_fscore_support(y_test, y_predict, average='binary')

print('Precision :', metrics[0])

print('Recall :', metrics[1])

print('Medida F1 :', metrics[2])

print('Error cuadratico medio: ', mean_squared_error(y_test, y_predict))

# Create a random forest Classifier. By convention, clf means 'Classifier'

clf = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators=1)

clf.fit(X_train, y_train)
indice_tree = 0 # Which tree we want to plot

fig, axes = plt.subplots(figsize = (20,20), dpi=800)

tree.plot_tree(clf.estimators_[indice_tree],

               feature_names = list(CATEGORIES.keys()), 

               class_names=['edible', 'poisonous'],

               filled = True,

               rounded = True);
y_forrest_predict = clf.predict(X_test)

metrics = precision_recall_fscore_support(y_test, y_forrest_predict, average='binary')

print('Precision :', metrics[0])

print('Recall :', metrics[1])

print('Medida F1 :', metrics[2])

print('Error cuadratico medio: ', mean_squared_error(y_test, y_forrest_predict))
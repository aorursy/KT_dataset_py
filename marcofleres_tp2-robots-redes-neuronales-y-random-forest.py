import numpy as np

import pandas as pd

import tensorflow as tf



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
DATAFILE="/kaggle/input/mushroom-classification/mushrooms.csv"

DATASET_SIZE=8124

np.set_printoptions(precision=3, suppress=True)

N_CLASSES=1 # Edible/Poisonous

N_INPUTS=22 # Columnas del CSV usadas para clasificar



full_dataset = tf.data.experimental.make_csv_dataset(DATAFILE, batch_size=1, label_name="class", shuffle=True)

# Ya que hay una sola clase "edible/poisonous", codificamos el label de los rows como 0/1. Luego codificaremos los features.

full_dataset = full_dataset.map(lambda features, label: (features, 0 if label=="e" else 1))



for features, label in full_dataset.take(1):

    print("Label:", label.numpy())

    tf.print(features)
train_data = full_dataset.take(round(DATASET_SIZE/3*2))

test_data = full_dataset.skip(round(DATASET_SIZE/3*2)).take(round(DATASET_SIZE/3)-1)





print(tf.data.experimental.cardinality(train_data).numpy())

print(tf.data.experimental.cardinality(test_data).numpy())
from tensorflow import feature_column



feature_columns=[]



VOCABULARY={

    'cap-shape': ['b', 'c', 'x', 'f', 'k', 's'],

    'cap-surface': ['f', 'g', 'y', 's'],

    'cap-color': ['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'],

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

    'stalk-color-above-ring': ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'],

    'stalk-color-below-ring': ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'],

    'veil-type': ['p', 'u'],

    'veil-color': ['n', 'o', 'w', 'y'],

    'ring-number': ['n', 'o', 't'],

    'ring-type': ['c', 'e', 'f', 'l', 'n', 'p', 's', 'z'],

    'spore-print-color': ['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y'],

    'population': ['a', 'c', 'n', 's', 'v', 'y'],

    'habitat': ['g', 'l', 'm', 'p', 'u', 'w', 'd'],

}



for header in VOCABULARY:

    feature_columns.append(feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list(header, VOCABULARY[header])))



feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

for features, label in train_data.take(1):

    print("Label:", label.numpy())

    print("Parametros:")

    tf.print(features)

    print("Parametros codificados:", feature_layer(features))
from tensorflow.keras import layers

from keras.optimizers import SGD



model = tf.keras.Sequential([

  #tf.keras.Input(shape=(126)),

  feature_layer,

  layers.Dense(20, activation='relu', kernel_initializer='he_normal', name="layer1"),

  layers.Dense(20, activation='relu', kernel_initializer='he_normal', name="layer1"),

  layers.Dense(1, name="output")

])



model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='mse', metrics=['accuracy'])

#print(model.summary())



#train_data = train_data.batch(100)

model.fit(train_data.batch(100), epochs=10, batch_size=100, verbose=1)
for features, label in test_data.take(20):

    print("Label/Prediction:", label.numpy(), model.predict_classes(features)[0][0])
from math import sqrt



# Evaluamos usando el conjunto de datos de test

loss, accuracy = model.evaluate(test_data.batch(1))

print('MSE: %.3f' % (loss))

print('Accuracy: %.3f' % (accuracy))



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



#predictions = test_data.map(lambda features, labels: (labels, tf.map_fn(lambda f:model.predict_classes(f), features)))

predictionPairs = []



for features, label in test_data.take(10):

    predictionPairs.append( (label.numpy(), model.predict_classes(features)[0][0]) )



y_true, y_pred = zip(*predictionPairs)

    

print({

    "accuracy": accuracy_score(y_true, y_pred),

    "precision": precision_score(y_true, y_pred),

    "recall": recall_score(y_true, y_pred),

    "f1": f1_score(y_true, y_pred)

})
dataPd = pd.read_csv(DATAFILE)

dataPd.head()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder



Y = dataPd['class']

X = dataPd.drop(['class'], axis=1)



# Necesitamos codificar los features como números para alimentar el Random Forest

X = OneHotEncoder().fit_transform(X).toarray()



train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=0.33, random_state=0)
from sklearn.ensemble import RandomForestClassifier



RF_model = RandomForestClassifier(bootstrap=True, n_estimators=50)

RF_model.fit(train_X, train_y)
from sklearn.metrics import accuracy_score, recall_score, f1_score



real = list(test_y.tolist())

RF_predictions = list(RF_model.predict(test_X))



print({

    "accuracy": accuracy_score(real, RF_predictions),

    "precision": precision_score(real, RF_predictions, pos_label="e"),

    "recall": recall_score(real, RF_predictions, pos_label="e"),

    "f1": f1_score(real, RF_predictions, pos_label="e")

})
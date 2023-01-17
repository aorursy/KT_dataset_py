import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt



import sqlite3



from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler



import tensorflow as tf
tf.__version__
# fix random seed for reproducibility

seed = 42

np.random.seed(seed)
connection = sqlite3.connect('../input/database.sqlite')

data = pd.read_sql_query(''' SELECT * FROM IRIS ''', connection)

print("Shape of data: {}".format(data.shape))
data.head(3)
data.info()
Y = data['Species']

X = data.drop(['Id', 'Species'], axis=1)

print("Shape of Input  features: {}".format(X.shape))

print("Shape of Output features: {}".format(Y.shape))
Y.value_counts()
lbl_clf = LabelEncoder()

Y_encoded = lbl_clf.fit_transform(Y)



#Keras requires your output feature to be one-hot encoded values.

Y_final = tf.keras.utils.to_categorical(Y_encoded)



print("Therefore, our final shape of output feature will be {}".format(Y_final.shape))
x_train, x_test, y_train, y_test = train_test_split(X, Y_final, test_size=0.25, random_state=seed, stratify=Y_encoded, shuffle=True)



print("Training Input shape\t: {}".format(x_train.shape))

print("Testing Input shape\t: {}".format(x_test.shape))

print("Training Output shape\t: {}".format(y_train.shape))

print("Testing Output shape\t: {}".format(y_test.shape))
std_clf = StandardScaler()

x_train_new = std_clf.fit_transform(x_train)

x_test_new = std_clf.transform(x_test)
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(10, input_dim=4, activation=tf.nn.relu, kernel_initializer='he_normal', 

                                kernel_regularizer=tf.keras.regularizers.l2(0.01)))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Dense(7, activation=tf.nn.relu, kernel_initializer='he_normal', 

                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001)))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Dense(5, activation=tf.nn.relu, kernel_initializer='he_normal', 

                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001)))

model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



iris_model = model.fit(x_train_new, y_train, epochs=700, batch_size=7)
model.evaluate(x_test_new, y_test)
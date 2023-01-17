# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

from time import time

import cv2 as cv2

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import tensorflow as tf

from sklearn.metrics import accuracy_score,  auc, roc_curve

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier



%matplotlib inline
# Input data files are available in the "../input/" directory.

print(os.listdir("../input"))

#print(os.path.getsize("../input/train_v2"))

root_train = "../input/train-jpg"

print(os.path.getsize("../input/train-jpg"))



# Any results you write to the current directory are saved as output.
labels_train = pd.DataFrame.from_csv("../input/train_v2.csv") # read the labels
cloudy = labels_train[labels_train.tags=="cloudy"]

primary = labels_train[labels_train.tags=="clear primary"]
def sample(y, k):

    if y == 0:

        return mpimg.imread(os.path.join(root_train, cloudy.index[k]+'.jpg'))

    elif y == 1:

        return mpimg.imread(os.path.join(root_train, primary.index[k]+'.jpg'))

    else:

        raise ValueError

        

sample(0,0).shape
f, ax = plt.subplots(4, 2, figsize=(5, 10))

for k in range(4):

    for y in range(2):

        ax[k][y].imshow(sample(y, k))
y_train_list = list()

data_train = np.zeros(((2*len(cloudy)), 32, 32, 4))

for i in range(len(cloudy)):

    data_train[i]= cv2.resize(sample(0,i),(32,32))

    y_train_list.append(0)    

for j in range(len(cloudy), 2*len(cloudy)):

    data_train[j]= cv2.resize(sample(1,j),(32,32))

    y_train_list.append(1)

len(y_train_list)
y_full = np.array(y_train_list)

y_full.shape
x_train, x_test, y_train, y_test = train_test_split(data_train, y_full, test_size=.25)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

print("\ndistribution of train classes")

print(pd.Series(y_train).value_counts())

print("\ndistribution of test classes")

print(pd.Series(y_test).value_counts())
def get_conv_model_A(num_classes, img_size=32, compile=True):

    tf.reset_default_graph()

    tf.keras.backend.clear_session()

    print("using",num_classes,"classes")

    inputs = tf.keras.Input(shape=(img_size,img_size,4), name="input_1")

    layers = tf.keras.layers.Conv2D(15,(3,3), activation="relu")(inputs)

    layers = tf.keras.layers.Flatten()(layers)

    layers = tf.keras.layers.Dense(16, activation=tf.nn.relu)(layers)

    layers = tf.keras.layers.Dropout(0.2)(layers)

    predictions = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax, name="output_1")(layers)

    model = tf.keras.Model(inputs = inputs, outputs=predictions)

    if compile:

        model.compile(optimizer='adam',

                      loss='sparse_categorical_crossentropy',

                      metrics=['accuracy'])

    return model
num_classes = len(np.unique(y_full))

model = get_conv_model_A(num_classes)
weights = model.get_weights()

for i in weights:

    print(i.shape)
num_classes = len(np.unique(y_full))



def train(model, batch_size, epochs, model_name=""):

    # Helper para el entrenamiento

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/"+model_name+"_"+"{}".format(time()))

    model.reset_states()

    model.fit(x_train, y_train, epochs=epochs, callbacks=[tensorboard],

              batch_size=batch_size,

              validation_data=(x_test, y_test))

    metrics = model.evaluate(x_test, y_test)

    return {k:v for k,v in zip (model.metrics_names, metrics)}
def get_conv_model_B(num_classes = 2, filters = 60, img_size=32, compile=True):

    tf.reset_default_graph()

    tf.keras.backend.clear_session()

    print("using",num_classes,"classes")

    inputs = tf.keras.Input(shape=(img_size,img_size,4), name="input_1")

    layers = tf.keras.layers.Conv2D(15,(5,5), activation="relu")(inputs)

    layers = tf.keras.layers.MaxPool2D((2,2))(layers)

    layers = tf.keras.layers.Conv2D(filters,(5,5), activation="relu")(layers)

    layers = tf.keras.layers.Flatten()(layers)

    layers = tf.keras.layers.Dense(16, activation=tf.nn.relu)(layers)

    layers = tf.keras.layers.Dropout(0.2)(layers)

    predictions = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax, name="output_1")(layers)

    

    model = tf.keras.Model(inputs = inputs, outputs=predictions)



    if compile:

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005),

                      

                      loss='sparse_categorical_crossentropy',

                      metrics=['accuracy'])

    return model



def get_conv_model_C(num_classes = 2, filters = 60, img_size=32, compile=True):

    tf.reset_default_graph()

    tf.keras.backend.clear_session()

    print("using",num_classes,"classes")    

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(15,(5,5), activation='relu', input_shape=(img_size,img_size,4)))

    model.add(tf.keras.layers.MaxPool2D((2,2)))

    model.add(tf.keras.layers.Conv2D(filters,(5,5), activation="relu"))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))

    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax, name="output_1"))



    if compile:

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005),

                      loss='sparse_categorical_crossentropy',

                      metrics=['accuracy'])

    return model
model = get_conv_model_B(num_classes)

model.summary()

train(model, batch_size=32, epochs=10, model_name="model_B")


modelC = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=get_conv_model_C, epochs=10)

pipe = Pipeline([('modelC', modelC)])

param_grid = {'modelC__filters': [60,20,100]}

search = GridSearchCV(pipe, param_grid, scoring = "accuracy", n_jobs=1, verbose=0)
search.fit(x_train, y_train)
print(search.best_estimator_.score(x_test, y_test))

print(search.best_params_)
pd.DataFrame(search.cv_results_)[['params', 'mean_test_score', 'std_test_score']]
model2 = get_conv_model_C(num_classes)

model2.summary()

train(model2, batch_size=32, epochs=10, model_name="model_C")
y_pred_keras = model2.predict(x_test)



y_pred_list=[]

for i in y_pred_keras:

    y_pred_list.append(np.argmax(i))

y_pred = np.array(y_pred_list)



fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred)



auc_keras = auc(fpr_keras, tpr_keras)
# Supervised transformation based on random forests

rf = RandomForestClassifier(max_depth=4, n_estimators=10)

rf.fit(x_train.reshape(-1,4096), y_train)



y_pred_rf = rf.predict_proba(x_test.reshape(-1,4096))[:, 1]

fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_pred_rf)

auc_rf = auc(fpr_rf, tpr_rf)
plt.figure(1)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))

plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.legend(loc='best')

plt.show()

# Zoom in view of the upper left corner.

plt.figure(2)

plt.xlim(0, 0.2)

plt.ylim(0.8, 1)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))

plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve (zoomed in at top left)')

plt.legend(loc='best')

plt.show()
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



from time import time

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



from IPython.display import Image

from multiprocessing import Pool

from scipy.stats import multivariate_normal

import cv2

import tensorflow as tf

tf.__version__



from IPython.display import Image

from sklearn.model_selection import train_test_split

%matplotlib inline
root = "../input"

root_test = root + '/test/test'

root_train = root + '/train/train'

print(os.listdir(root))
identify = pd.DataFrame.from_csv(root + '/train.csv')
total_test = os.listdir(root_test)

total_train = os.listdir(root_train)

len(total_train)
identify[identify.index=='6ba575bb5788a9b77a4cef2fb4fdf24d.jpg'].has_cactus.values[0]

#identify[0:1].has_cactus.values[0]

identify[identify.has_cactus==1].index[0]
total_data = +len(total_test)+len(total_train)
total_cactus = len(identify[identify.has_cactus.values==1])

total_no_cactus = len(identify[identify.has_cactus.values==0])

total_no_cactus
print('{} Total test'.format(len(total_test)))

print('{} Total train'.format(len(total_train)))
def sample(y, k):

    if y == 0:

        return mpimg.imread(os.path.join(root_train, identify[identify.has_cactus==0].index[k]))

    elif y == 1:

        return mpimg.imread(os.path.join(root_train, identify[identify.has_cactus==1].index[k]))

    else:

        raise ValueError
sample(0, 0).shape
f, ax = plt.subplots(4, 2, figsize=(10, 10))

for k in range(4):

    for y in range(2):

        ax[k][y].imshow(sample(y, k))
sample(0,1).shape
y_total = []

x_total = np.zeros((total_no_cactus+total_no_cactus,16,16,3))

for i in range(total_no_cactus):

    x_total[i] = cv2.resize(sample(0,i),(16,16))

    y_total.append(0)

for j in range(total_no_cactus):

    x_total[j+total_no_cactus] = cv2.resize(sample(1,j),(16,16))

    y_total.append(1)
len(y_total)
y_total_np=np.array(y_total)

y_total_np[total_no_cactus-1]
x_train, x_test, y_train, y_test = train_test_split(x_total, y_total_np, test_size=.25)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

print("\ndistribution of train classes")

print(pd.Series(y_train).value_counts())

print("\ndistribution of test classes")

print(pd.Series(y_test).value_counts())
def get_conv_model_A(num_classes, img_size=16, compile=True):

    tf.reset_default_graph()

    tf.keras.backend.clear_session()

    print("using",num_classes,"classes")

    inputs = tf.keras.Input(shape=(img_size,img_size,3), name="input_1")

    layers = tf.keras.layers.Conv2D(12,(3,3), activation="relu")(inputs)

    layers = tf.keras.layers.Flatten()(layers)

    layers = tf.keras.layers.Dense(15, activation=tf.nn.relu)(layers)

    layers = tf.keras.layers.Dropout(0.2)(layers)

    predictions = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax, name="output_1")(layers)

    model = tf.keras.Model(inputs = inputs, outputs=predictions)

    if compile:

        model.compile(optimizer='adam',

                      loss='sparse_categorical_crossentropy',

                      metrics=['accuracy'])

    return model
model = get_conv_model_A(2)
weights = model.get_weights()

for i in weights:

    print(i.shape)
initial_w0 = model.get_weights()[0].copy()
y_test.shape, y_train.shape, x_test.shape, x_train.shape
num_classes = len(np.unique(y_total_np))



def train(model, batch_size, epochs, model_name=""):

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/"+model_name+"_"+"{}".format(time()))

    model.reset_states()

    model.fit(x_train, y_train, epochs=epochs, callbacks=[tensorboard],

              batch_size=batch_size,

              validation_data=(x_test, y_test))

    metrics = model.evaluate(x_test, y_test)

    return {k:v for k,v in zip (model.metrics_names, metrics)}
model = get_conv_model_A(num_classes)

model.summary()

train(model, batch_size=32, epochs=20, model_name="model_A")
def get_conv_model_B(num_classes=2 ,entr=60,convol1=70,convol2=75, img_size=16, compile=True):

    tf.reset_default_graph()

    tf.keras.backend.clear_session()

    print("using",num_classes,"classes")

    model = tf.keras.models.Sequential()

    

    model.add(tf.keras.layers.Conv2D(convol1,(3,3), activation="relu",input_shape=(img_size,img_size,3)))

    model.add(tf.keras.layers.MaxPool2D((2,2)))

    model.add(tf.keras.layers.Conv2D(convol2,(3,3), activation="relu"))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(entr, activation=tf.nn.relu))

    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax, name="output_1"))

    

    if compile:

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005),

                      loss='sparse_categorical_crossentropy',

                      metrics=['accuracy'])

    return model
model = get_conv_model_B(num_classes, 70, 7, 11)

model.summary()

train(model, batch_size=32, epochs=20, model_name="model_B")
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPClassifier

#entr(60,70,75), convol1(5,7,9), convol2(11,12,15)



model2 = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=get_conv_model_B,epochs=10)



pipe = Pipeline([('model2', model2)])



param_grid = {'model2__entr': [60,70,], 

              'model2__convol1': [5,7,],}

search = GridSearchCV(pipe, param_grid, scoring="accuracy", n_jobs=1, verbose=1)
search.fit(x_train, y_train)
search.best_estimator_.score(x_test, y_test)
search.best_params_
pd.DataFrame(search.cv_results_).columns
pd.DataFrame(search.cv_results_)[['params', 'mean_test_score', 'std_test_score']]
model3 = get_conv_model_B(2)

model3.summary()

train(model3, batch_size=32, epochs=10, model_name="model3")
from sklearn.metrics import roc_curve

y_pred_keras = model3.predict(x_test)

y_pred_list=[]

for i in y_pred_keras:

    y_pred_list.append(np.argmax(i))

y_pred = np.array(y_pred_list)

fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred)
from sklearn.metrics import auc

auc_keras = auc(fpr_keras, tpr_keras)

from sklearn.ensemble import RandomForestClassifier

# Supervised transformation based on random forests

rf = RandomForestClassifier(max_depth=4, n_estimators=10)

rf.fit(x_train.reshape(-1,768), y_train)



y_pred_rf = rf.predict_proba(x_test.reshape(-1,768))[:, 1]

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
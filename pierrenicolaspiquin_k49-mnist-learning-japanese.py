# Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Machine learning
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, Dense, Dropout, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy
from keras.losses import categorical_crossentropy
from keras.models import Sequential
# Paths
input_path = os.path.join('..', 'input')
classmap_path = os.path.join(input_path, 'k49_classmap.csv')

k49_train_imgs_path = os.path.join(input_path, 'k49-train-imgs.npz')
k49_train_labels_path = os.path.join(input_path, 'k49-train-labels.npz')
k49_test_imgs_path = os.path.join(input_path, 'k49-test-imgs.npz')
k49_test_labels_path = os.path.join(input_path, 'k49-test-labels.npz')

# Learning
n_classes = 49
learning_rate = 0.001
image_shape = (28, 28, 1)
n_epochs = 25
# Classmap loading : pandas dataframe that links an index to an hiragana
# Here 0 is a 'a', 1 a 'i', 2 a 'u', ...
k49_classmap = pd.read_csv(classmap_path)
k49_classmap.head()
# Data loading
train_imgs = np.load(k49_train_imgs_path)['arr_0']
train_labels = np.load(k49_train_labels_path)['arr_0']
test_imgs = np.load(k49_test_imgs_path)['arr_0']
test_labels = np.load(k49_test_labels_path)['arr_0']
# Data visualization : let's plot some hiraganas
n = 7
fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(8, 8))
for i in range(n**2):
    ax = axs[i // n, i % n]
    ax.imshow(train_imgs[i], cmap='Greys')
    ax.axis('off')
plt.tight_layout()
plt.show()
# Let's plot the data repartition between all our classes
train_labels_s = pd.Series(train_labels)
train_labels_s.value_counts(sort=False).plot.bar()
# here I still need to find a way to enlarge this plot !
# Using expand_dims to get a nominal deep learning format for all images
# (28, 28) --> (28, 28, 1)
train_imgs = np.expand_dims(train_imgs, axis=-1)
test_imgs = np.expand_dims(test_imgs, axis=-1)
# creation of a training set and a validation one
x_train, x_val, y_train, y_val = train_test_split(train_imgs, train_labels, test_size=0.10)
# One hot encoding util function
def one_hot_encoding(y):
    y_res = np.zeros((len(y), n_classes))
    for i in range(len(y)):
        y_res[i][y[i]] = 1
    return y_res
# Get the labels in a one hot encoded version
y_train = one_hot_encoding(y_train)
y_val = one_hot_encoding(y_val)
# Model definition - simple CNN model
def define_cnn_model(input_shape, output_nodes):
    model = Sequential()

    model.add(Conv2D(32, (5, 5), strides=(1, 1), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(96, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Conv2D(96, (3, 3), strides=(1, 1), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(output_nodes, activation='softmax'))
    
    return model

model = define_cnn_model(image_shape, n_classes)
model.compile(optimizer=Adam(learning_rate), loss=categorical_crossentropy, metrics=[categorical_accuracy])
# Let's look at our model
model.summary()
training_recap = model.fit(x_train, y_train, epochs=n_epochs, validation_data=(x_val, y_val), batch_size=128)
history = training_recap.history
e = [i for i in range(1, n_epochs+1)]
loss = history['loss']
val_loss = history['val_loss']

plt.plot(e, loss, val_loss)
plt.title('Training and validation losses')
plt.show()
# Prediction on test set
print("Categorical accuracy : {:.3f}".format(model.evaluate(test_imgs, one_hot_encoding(test_labels))[1]))
y_pred = model.predict(test_imgs)
y_pred = np.argmax(y_pred,axis=1)
# Let's visualize some predictions
n = 5
fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(15, 15))
for i in range(n**2):
    ax = axs[i // n, i % n]
    ax.imshow(test_imgs[i, :, :, 0], cmap='Greys')
    ax.set_title('Class: {}, Predicted: {}'.format(test_labels[i], y_pred[i]))
plt.show()
from sklearn.metrics import classification_report

print(classification_report(test_labels, y_pred, target_names=k49_classmap['char']))
# Definition of the penalization weights
train_labels_s = pd.Series(train_labels)
n_sample_per_class = train_labels_s.value_counts(sort=False)
max_sample_per_class = n_sample_per_class.max()
weights = np.array([max_sample_per_class / w for w in n_sample_per_class])
weights = weights / weights.max()
# print(weights)
# custom loss function
"""
A weighted version of categorical_crossentropy for keras (2.0.6). This lets you apply a weight to unbalanced classes.
@url: https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
@author: wassname
"""
from keras import backend as K
def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

custom_categorical_crossentropy = weighted_categorical_crossentropy(weights)
# Redefinintion of our model
model_custom_loss = define_cnn_model(image_shape, n_classes)
model_custom_loss.compile(optimizer=Adam(learning_rate), loss=custom_categorical_crossentropy, metrics=[categorical_accuracy])
training_recap_c = model_custom_loss.fit(x_train, y_train, epochs=n_epochs, validation_data=(x_val, y_val), batch_size=128)
# Prediction on test set
print("Categorical accuracy : {:.3f}".format(model_custom_loss.evaluate(test_imgs, one_hot_encoding(test_labels))[1]))
y_pred = model_custom_loss.predict(test_imgs)
y_pred = np.argmax(y_pred,axis=1)
print(classification_report(test_labels, y_pred, target_names=k49_classmap['char']))
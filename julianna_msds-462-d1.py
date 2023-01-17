# Helper libraries
from datetime import date, timedelta, datetime
from packaging import version
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from bokeh.plotting import output_notebook, figure, show
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import numpy as np
import pandas as pd

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from keras import models, layers, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dropout, Flatten, Input, Dense
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
train = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')

print('train:\t{}'.format(train.shape))

train.head(3)
validation = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')

print('validation:\t{}'.format(validation.shape))

validation.head(3)
# Assign features & label in training set
X = np.array(train.drop(columns=['label']))
y = np.array(train['label'])

print('X:\t{}'.format(X.shape))
print('y:\t{}'.format(y.shape))
# Assign features & label in validation set
validation_x = np.array(validation.drop(columns=['label']))
validation_y = np.array(validation['label'])

print('validation_x:\t{}'.format(validation_x.shape))
print('validation_y:\t{}'.format(validation_y.shape))
# Split training data into train and test
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)

print('train_x:\t{}'.format(train_x.shape))
print('test_x:\t{}'.format(test_x.shape))
print('train_y:\t{}'.format(train_y.shape))
print('test_y:\t{}'.format(test_y.shape))
# Encode labels across train, test & validation sets
train_y_encoded = to_categorical(train_y)
test_y_encoded = to_categorical(test_y)
val_y_encoded = to_categorical(validation_y)

print('train_y_encoded:\t{}'.format(train_y_encoded.shape))
print('test_y_encoded:\t{}'.format(test_y_encoded.shape))
print('val_y_encoded:\t{}'.format(val_y_encoded.shape))
# Normalize

train_x = np.array(train_x)/255
test_x = np.array(test_x)/255

train_y_encoded = np.array(train_y_encoded)
test_y_encoded = np.array(test_y_encoded)

validation_x = np.array(validation_x)
val_y_encoded = np.array(val_y_encoded)


print('train_x:\t{}'.format(train_x.shape))
print('train_y_encoded:\t{}'.format(train_y_encoded.shape))
print('test_x:\t{}'.format(test_x.shape))
print('test_y_encoded:\t{}'.format(test_y_encoded.shape))
print('validation_x:\t{}'.format(validation_x.shape))
print('val_y_encoded:\t{}'.format(val_y_encoded.shape))

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
%matplotlib inline
fig = plt.figure(figsize = (15, 9))

for i in range(50):
    plt.subplot(5, 10, 1+i)
    plt.title(class_names[train_y[i]], fontsize=14)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_x[i].reshape(28,28), cmap='binary')
# Simple Dense Neural Network

input_size = 784
output_size = 10

sdnn = models.Sequential()
sdnn.add(layers.Dense(input_size, activation='relu'))
sdnn.add(layers.Dense(output_size, activation='softmax'))
sdnn.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
start = datetime.now()

sdnn_history = sdnn.fit(train_x,
                 train_y_encoded,
                 epochs=20,
                 batch_size=512,
                 validation_data=(test_x, test_y_encoded)
                 )

end = datetime.now()
sdnn_run_time = end - start

sdnn.save('sdnn.h5')
p = figure(plot_width=800, plot_height=400, title='Simple Dense Neural Network: Training and Validation Accuracy', x_axis_label='Epoch', y_axis_label='Accuracy')
x = np.arange(1,21,1)
p.line(x,sdnn_history.history['accuracy'], legend_label='Train Accuracy', line_width=3)
p.line(x,sdnn_history.history['val_accuracy'], legend_label='Validation Accuracy', color='green', line_width=3)
p.legend.location = "top_left"
show(p)
print(sdnn_run_time)
# Evaluate prediction power of the simple dense neural network model
y_pred = sdnn.predict_classes(validation_x)

# Confusion matrix
cm = pd.DataFrame(confusion_matrix(y_pred, validation_y), index=class_names, columns=class_names)
cm
plt.matshow(cm, cmap=plt.cm.gray)
plt.xlabel("Predicted Classes")
plt.ylabel("Actual Classes")
plt.show();
cr = classification_report(y_pred, validation_y,target_names=class_names, output_dict=True)
cr_df = pd.DataFrame(cr).transpose()
cr_df = np.round(cr_df,2)
cr_df
# Dense Neural Network: Two Hidden Layers with 20% Regularization

hdnn = models.Sequential()
hdnn.add(layers.Dense(input_size, activation='relu'))
hdnn.add(layers.Dropout(0.2))
hdnn.add(layers.Dense(392, activation='relu'))
hdnn.add(layers.Dropout(0.2))
hdnn.add(layers.Dense(output_size, activation='softmax'))

hdnn.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
start = datetime.now()

hdnn_history = hdnn.fit(train_x,
                 train_y_encoded,
                 epochs=20,
                 batch_size=512,
                 validation_data=(test_x, test_y_encoded)
                 )

end = datetime.now()
hdnn_run_time = end - start

hdnn.save('hdnn.h5')
p = figure(plot_width=800, plot_height=400, title='Dense Neural Network with two hidden layers, 20% regularization: Training and Validation Accuracy', x_axis_label='Epoch', y_axis_label='Accuracy')
x = np.arange(1,21,1)
p.line(x,hdnn_history.history['accuracy'], legend_label='Train Accuracy', line_width=3)
p.line(x,hdnn_history.history['val_accuracy'], legend_label='Validation Accuracy', color='green', line_width=3)
p.legend.location = "top_left"
show(p)
print(hdnn_run_time)
# Evaluate prediction power of the simple dense neural network model
y_pred = hdnn.predict_classes(validation_x)

# Confusion matrix
cm = pd.DataFrame(confusion_matrix(y_pred, validation_y), index=class_names, columns=class_names)
cm
plt.matshow(cm, cmap=plt.cm.gray)
plt.xlabel("Predicted Classes")
plt.ylabel("Actual Classes")
plt.show();
cr = classification_report(y_pred, validation_y,target_names=class_names, output_dict=True)
cr_df = pd.DataFrame(cr).transpose()
cr_df = np.round(cr_df,2)
cr_df
# pre-process input data for CNN architecture

image_shape=(28,28,1)

train_x_cnn = np.array(train_x).reshape(train_x.shape[0],*image_shape)
test_x_cnn = np.array(test_x).reshape(test_x.shape[0],*image_shape)
val_x_cnn = np.array(validation_x).reshape(validation_x.shape[0],*image_shape)


print('train_x_cnn:\t{}'.format(train_x_cnn.shape))
print('test_x_cnn:\t{}'.format(test_x_cnn.shape))
print('val_x_cnn:\t{}'.format(val_x_cnn.shape))
# Simple Convolutional Neural Network

scnn = models.Sequential()
scnn.add(layers.Conv2D(28, kernel_size=(3,3),activation='relu',padding='same',input_shape=(28,28,1)))
scnn.add(layers.MaxPooling2D((2, 2),padding='same'))
scnn.add(layers.Flatten())
scnn.add(layers.Dense(10, activation='softmax'))

scnn.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
start = datetime.now()

scnn_history = scnn.fit(train_x_cnn,
                 train_y_encoded,
                 epochs=20,
                 batch_size=512,
                 validation_data=(test_x_cnn, test_y_encoded)
                 )

end = datetime.now()
scnn_run_time = end - start

scnn.save('scnn.h5')


p = figure(plot_width=800, plot_height=400, title='Simple Convolutional Neural Network: Train and Validation Accuracy', x_axis_label='Epoch', y_axis_label='Accuracy')
x = np.arange(1,56,1)
p.line(x,scnn_history.history['accuracy'], legend_label='Train Accuracy', line_width=3)
p.line(x,scnn_history.history['val_accuracy'], legend_label='Validation Accuracy', color='green', line_width=3)
p.legend.location = "bottom_right"
show(p)
print(scnn_run_time)
# Evaluate prediction power of the model
y_pred = scnn.predict_classes(val_x_cnn)

# Confusion matrix
cm = pd.DataFrame(confusion_matrix(y_pred, validation_y), index=class_names, columns=class_names)
cm
plt.matshow(cm, cmap=plt.cm.gray)
plt.xlabel("Predicted Classes")
plt.ylabel("Actual Classes")
plt.show();
cr = classification_report(y_pred, validation_y,target_names=class_names, output_dict=True)
cr_df = pd.DataFrame(cr).transpose()
cr_df = np.round(cr_df,2)
cr_df
# Convolutional Neural Network: Two Hidden Layers with 20% Regularization

hcnn = models.Sequential()
hcnn.add(layers.Conv2D(28, kernel_size=(3,3),activation='relu',padding='same',input_shape=(28,28,1)))
hcnn.add(layers.MaxPooling2D((2, 2),padding='same'))
hcnn.add(layers.Dropout(0.2))

hcnn.add(layers.Conv2D(56, kernel_size=(3,3),activation='relu',padding='same',input_shape=(28,28,1)))
hcnn.add(layers.MaxPooling2D((2, 2),padding='same'))
hcnn.add(layers.Dropout(0.2))

hcnn.add(layers.Flatten())
hcnn.add(layers.Dense(10, activation='softmax'))

hcnn.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
start = datetime.now()

hcnn_history = hcnn.fit(train_x_cnn,
                 train_y_encoded,
                 epochs=20,
                 batch_size=512,
                 validation_data=(test_x_cnn, test_y_encoded)
                 )

end = datetime.now()
hcnn_run_time = end - start

hcnn.save('hcnn.h5')
p = figure(plot_width=800, plot_height=400, title='Convolutional Neural Network with two hidden layer, max pooling and 20% dropout: Train and Validation Accuracy', x_axis_label='Epoch', y_axis_label='Accuracy')
x = np.arange(1,56,1)
p.line(x,hcnn_history.history['accuracy'], legend_label='Train Accuracy', line_width=3)
p.line(x,hcnn_history.history['val_accuracy'], legend_label='Validation Accuracy', color='green', line_width=3)
p.legend.location = "bottom_right"
show(p)
print(hcnn_run_time)
# Evaluate prediction power of the model
y_pred = hcnn.predict_classes(val_x_cnn)

# Confusion matrix
cm = pd.DataFrame(confusion_matrix(y_pred, validation_y), index=class_names, columns=class_names)
cm
plt.matshow(cm, cmap=plt.cm.gray)
plt.xlabel("Predicted Classes")
plt.ylabel("Actual Classes")
plt.show();
cr = classification_report(y_pred, validation_y,target_names=class_names, output_dict=True)
cr_df = pd.DataFrame(cr).transpose()
cr_df = np.round(cr_df,2)
cr_df
benchmark = pd.DataFrame({'model': ['Simple Dense Neural Network', 'Dense Neural Network. Two hidden layers. 20% Dropout',\
                       'Simple Convolutional Neural Network', 'Convolutional Neural Network. Two hidden layers. 20% Dropout'],
             'processing_time': [sdnn_run_time, hdnn_run_time, scnn_run_time, hcnn_run_time],
             'training_accuracy': [0.96, 0.59, 0.91, 0.90],
             'validation_accuracy': [0.89, 0.45,0.90, 0.91],
             'testing_accuracy': [0.89,0.52,0.82,0.88]})
print(benchmark)

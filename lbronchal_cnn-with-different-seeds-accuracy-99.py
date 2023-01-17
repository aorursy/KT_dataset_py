import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter("ignore", category=FutureWarning)
data_dir = "../input/Sign-language-digits-dataset/"
X = np.load(data_dir + "X.npy")
y = np.load(data_dir + "Y.npy")
plt.figure(figsize=(10,10))
for i in range(0, 9):
    plt.subplot(330 + 1 + i)
    plt.tight_layout()
    plt.imshow(X[i], cmap=plt.get_cmap('gray'))
    plt.axis('off')
    plt.title(y[i].argmax())    
plt.show()
X.shape
X.min(), X.max()
num_cases = np.unique(y.argmax(axis=1), return_counts=True) 
plt.title("Number of cases")
plt.xticks(num_cases[0])
plt.bar(num_cases[0], num_cases[1])
plt.show()
import tensorflow as tf
import random as rn
from keras import backend as K

os.environ['PYTHONHASHSEED'] = '0'

SEED = 1
np.random.seed(SEED)
rn.seed(SEED)
img_rows , img_cols, img_channel = 64, 64, 1
target_size = (img_rows, img_cols)
target_dims = (img_rows, img_cols, img_channel) 
n_classes = 10
val_frac = 0.2
batch_size = 64
X = X.reshape(X.shape[0], img_rows, img_cols, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=val_frac, stratify=y, random_state=SEED)
from keras.layers import Dropout, Flatten, Dense, Convolution2D, MaxPooling2D, Input, BatchNormalization
from keras.models import Sequential, Model 
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from tensorflow import set_random_seed
def initialize_nn_seed(seed):
    np.random.seed(seed)
    rn.seed(seed)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.set_random_seed(seed)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)    
def create_model(seed):
    initialize_nn_seed(seed)    
    
    model = Sequential()
    model.add(Convolution2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(64, 64, 1)))
    model.add(Convolution2D(16, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(Convolution2D(32, kernel_size=(3, 3), strides=(2, 2), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, kernel_size=(2, 2), strides=(1, 1), activation='relu'))
    model.add(Convolution2D(64, kernel_size=(2, 2), strides=(2, 2), activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-3), metrics=["accuracy"])
    return model
def train_model(model): 
    epochs = 40 
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
    checkpoint = ModelCheckpoint("weights-best.hdf5", monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    annealer = ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=1, factor=0.5, min_lr=0.00001)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=10*batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, checkpoint, annealer]
    )  
    model.load_weights("weights-best.hdf5")
    return history, model
datagen = ImageDataGenerator(rotation_range=10,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range = 0.1,
                             fill_mode="nearest")
        
train_generator = datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True, seed=SEED, subset="training")
model = create_model(seed=123456)
history, model = train_model(model)
scores = model.evaluate(X_test, y_test, verbose=0)
print("{}: {:.2f}%".format(model.metrics_names[1], scores[1]*100))
model = create_model(seed=1)
history, model = train_model(model)
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
scores = model.evaluate(X_test, y_test, verbose=0)
print("{}: {:.2f}%".format(model.metrics_names[1], scores[1]*100))
predicted_classes = model.predict_classes(X_test)
y_true = y_test.argmax(axis=1)
import scikitplot as skplt
skplt.metrics.plot_confusion_matrix(y_true, predicted_classes, normalize=True, figsize=(8, 8))
plt.show()
correct = np.where(predicted_classes == y_true)[0]
incorrect = np.where(predicted_classes != y_true)[0]
plt.figure(figsize=(8, 8))
for i, correct in enumerate(incorrect[:9]):
    plt.subplot(430+1+i)
    plt.imshow(X_test[correct].reshape(img_rows, img_cols), cmap='gray')
    plt.title("Pred: {} || Class {}".format(predicted_classes[correct], y_true[correct]))
    plt.axis('off')
    plt.tight_layout()
plt.show()
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_true, predicted_classes))
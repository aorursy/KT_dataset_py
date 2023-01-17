import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob 
from skimage.io import imread #read images from files
import os
import keras.backend as K
import tensorflow as tf
base_tile_dir = '../input/train/'
df = pd.DataFrame({'path': glob(os.path.join(base_tile_dir,'*.tif'))})
df['id'] = df.path.map(lambda x: x.split('/')[3].split(".")[0])
labels = pd.read_csv("../input/train_labels.csv")
df = df.merge(labels, on = "id")
df.head(10)
df0 = df[df.label == 0].sample(5000, random_state = 42)
df1 = df[df.label == 1].sample(5000, random_state = 42)
df = pd.concat([df0, df1], ignore_index=True).reset_index()
df = df[["path", "id", "label"]]
df.sample(10)
df['image'] = df['path'].map(imread)
df.sample(3)
import matplotlib.pyplot as plt

images = [(df['image'][0], df['label'][0]), 
          (df['image'][1], df['label'][1]),
          (df['image'][2], df['label'][2]),
          (df['image'][5000], df['label'][5000]),
          (df['image'][5001], df['label'][5001]),
          (df['image'][5002], df['label'][5002])]

fig, m_axs = plt.subplots(1, len(images), figsize = (20, 2))
#show the images and label them
for ii, c_ax in enumerate(m_axs):
    c_ax.imshow(images[ii][0])
    c_ax.set_title(images[ii][1])
input_images = np.stack(list(df.image), axis = 0)
input_images.shape
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.utils import np_utils


train_fraction = 0.8

encoder = LabelBinarizer()
y = encoder.fit_transform(df.label)
x = input_images

train_tensors, test_tensors, train_targets, test_targets =\
    train_test_split(x, y, train_size = train_fraction, random_state = 42)

val_size = int(0.5*len(test_tensors))

val_tensors = test_tensors[:val_size]
val_targets = test_targets[:val_size]
test_tensors = test_tensors[val_size:]
test_targets = test_targets[val_size:]
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from tensorflow import set_random_seed

set_random_seed(42)

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5)
checkpointer = ModelCheckpoint(filepath='weights.hdf5', 
                               verbose=1, save_best_only=True)
model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = 3, padding = 'same', activation = 'relu', input_shape = (96, 96, 3)))
model.add(Conv2D(filters = 16, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 16, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size = 3)) 

model.add(Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu')) 
model.add(Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu')) 
model.add(Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size = 3)) 

model.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size = 3))

model.add(Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation = 'elu'))
model.add(Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation = 'elu'))
model.add(Conv2D(filters = 256, kernel_size = 3, padding = 'same', activation = 'elu'))

model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))
#working AUC metric for keras from here: 
#https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/41015
def auc(y_true, y_pred):   
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)

#---------------------
# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N

#----------------
# PTA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P

from keras.optimizers import SGD
# sgd = SGD(lr = 0.0001, momentum = 0.9)
model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['accuracy'])

epochs = 15
model.fit(train_tensors, train_targets, 
          validation_data=(val_tensors, val_targets),
          epochs=epochs, batch_size=80, verbose=1, callbacks = [early_stopping, checkpointer])
model.load_weights('weights.hdf5')

cancer_predictions =  [model.predict(np.expand_dims(tensor, axis=0))[0][0] for tensor in test_tensors]

test_accuracy = 100*np.sum(np.round(cancer_predictions).astype('int32')==test_targets.flatten())/len(cancer_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
#AUC score
from sklearn.metrics import roc_auc_score
score = roc_auc_score(np.round(cancer_predictions).astype('int32'), test_targets)
score
base_tile_dir = '../input/test/'
test_df = pd.DataFrame({'path': glob(os.path.join(base_tile_dir,'*.tif'))})
test_df['id'] = test_df.path.map(lambda x: x.split('/')[3].split(".")[0])
test_df['image'] = test_df['path'].map(imread)
test_images = np.stack(test_df.image, axis = 0)
test_images.shape
predicted_labels =  [model.predict(np.expand_dims(tensor, axis=0))[0][0] for tensor in test_images]
predictions = np.array(predicted_labels)
test_df['label'] = predictions
submission = test_df[["id", "label"]]
submission.head()
#submission
submission.to_csv("submission.csv", index = False, header = True)
import os

import tensorflow as tf

from tensorflow import keras

import numpy as np

import matplotlib.pyplot as plt

import scikitplot as skplt

import pandas as pd

from tensorflow.keras import layers

from tensorflow.keras.models import Sequential

from tensorflow.keras import optimizers

from tensorflow.keras import losses

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics import multilabel_confusion_matrix

from tensorflow.keras.callbacks import LearningRateScheduler
train_set = '/kaggle/input/digit-recognizer/train.csv'
df_tr = pd.read_csv(train_set, sep=',')
df_tr.head()
labels = np.array(df_tr)[:,0]

labels.shape
labels
samples = np.array(df_tr)[:,1:]

samples.shape
samples
img_arr = np.reshape(samples , (labels.shape[0], 28, 28, 1))

img_arr.shape
plt.imshow(img_arr[100, :, :, 0], cmap=plt.cm.gray, interpolation='bilinear')
df_train, df_test = train_test_split(df_tr, train_size=0.85, test_size=0.15, random_state=42, shuffle = True)

df_train, df_val = train_test_split(df_train, train_size=0.8, test_size=0.2, random_state=42, shuffle = True)
train_count = df_train['label'].value_counts()

print("Total : ", np.sum(train_count))

train_count.plot(kind='bar', title='Train set Count')
val_count = df_val['label'].value_counts()

print("Total : ", np.sum(val_count))

val_count.plot(kind='bar', title='validation set Count')
test_count = df_test['label'].value_counts()

print("Total : ", np.sum(test_count))

test_count.plot(kind='bar', title='Test set Count')
lb_style = LabelBinarizer()

lb_results = lb_style.fit_transform(df_train["label"])

tr_labels = pd.DataFrame(lb_results, columns=lb_style.classes_)

tr_labels.head()
lb_style = LabelBinarizer()

lb_results = lb_style.fit_transform(df_val["label"])

val_labels = pd.DataFrame(lb_results, columns=lb_style.classes_)

val_labels.head()
tr_img_arr = np.reshape(np.array(df_train)[:,1:] , (np.sum(train_count), 28, 28, 1))

print(tr_img_arr.shape)



val_img_arr = np.reshape(np.array(df_val)[:,1:] , (np.sum(val_count), 28, 28, 1))

print(val_img_arr.shape)
Generator = ImageDataGenerator(rescale = 1./255)

train_Gen = Generator.flow(x=tr_img_arr,

                y=np.array(tr_labels),

                batch_size=32,

                shuffle=True,

                seed=1)



val_Gen = Generator.flow(x=val_img_arr,

                y=np.array(val_labels),

                batch_size=32,

                shuffle=True,

                seed=1)
model = Sequential()

model.add(layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(28,28,1), padding='same'))

model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))

model.add(layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))

model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

model.add(layers.Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))

model.add(layers.Flatten())

model.add(layers.Dense(84, activation = 'tanh', name='FC1'))

model.add(layers.Dense(10, activation = 'softmax', name='FC2'))

for layer in model.layers:

    layer.trainable = True
model.summary()
def step_decay(epoch, lr):

    return 0.00025*np.exp(-0.09*epoch)

Schedule = LearningRateScheduler(step_decay, verbose=1)
model.compile(loss = losses.CategoricalCrossentropy(),

              optimizer = optimizers.Adam(),

              metrics=['accuracy'])
History = model.fit(train_Gen, 

                    epochs=50,

                    verbose=1,

                    validation_data = val_Gen,

                    callbacks = [Schedule])
x = np.arange(1, 50, 2) 

y = 0.00025*np.exp(-0.09*x)

plt.plot(x, y)

plt.xlabel('epoch') 

plt.ylabel('learning rate') 

plt.show()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

t = f.suptitle('Model Performance', fontsize=12)

f.subplots_adjust(top=0.85, wspace=0.3)



epoch_list = list(range(1,51))

ax1.plot(epoch_list, History.history['accuracy'], label='Train Accuracy')

ax1.plot(epoch_list, History.history['val_accuracy'], label='Validation Accuracy')

ax1.set_xticks(np.arange(0, 51, 2))

ax1.set_ylabel('Accuracy Value')

ax1.set_xlabel('Epoch')

ax1.set_title('Accuracy')

l1 = ax1.legend(loc="best")



ax2.plot(epoch_list, History.history['loss'], label='Train Loss')

ax2.plot(epoch_list, History.history['val_loss'], label='Validation Loss')

ax2.set_xticks(np.arange(0, 51, 2))

ax2.set_ylabel('Loss Value')

ax2.set_xlabel('Epoch')

ax2.set_title('Loss')

l2 = ax2.legend(loc="best")
test_img_arr = np.reshape(np.array(df_test)[:,1:] , (np.sum(test_count), 28, 28, 1))

print(test_img_arr.shape)



lb_style = LabelBinarizer()

lb_results = lb_style.fit_transform(df_test["label"])

test_labels = pd.DataFrame(lb_results, columns=lb_style.classes_)



test_Gen = Generator.flow(x=test_img_arr,

                y=None,

                batch_size=32,

                shuffle=False)
test_Gen.reset()

y_pred = model.predict(x = test_Gen, verbose = 1)

y_pred
Y_pred = (y_pred > 0.5).astype(int)

Y_pred
conf_mat = multilabel_confusion_matrix(y_true = np.array(test_labels), y_pred=Y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

print('Confusion matrix:\n', conf_mat)
test_set = '/kaggle/input/digit-recognizer/test.csv'

df_test = pd.read_csv(test_set, sep=',')

df_test.head()
tst_img_arr = np.reshape(np.array(df_test) , (28000, 28, 28, 1))

print(tst_img_arr.shape)
test_Gen = Generator.flow(x=tst_img_arr,

                y=None,

                batch_size=32,

                shuffle=False)
test_Gen.reset()

y_pred = model.predict(x = test_Gen, verbose = 1)
pred = (y_pred > 0.5).astype(int)

pred
predictions = np.argmax(pred, axis=1)

predictions.shape
df_pred = pd.DataFrame({'ImageId':np.arange(1, 28001, 1),

                      'Label':predictions})

df_pred
df_pred.to_csv('/kaggle/working/output.csv', index = False)
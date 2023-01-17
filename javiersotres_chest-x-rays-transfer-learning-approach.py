import cv2

import os

import numpy as np

import matplotlib.pyplot as plt

import pickle

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.applications.vgg16 import VGG16



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score



from mlxtend.plotting import plot_confusion_matrix

from tensorflow.keras.utils import to_categorical
CATEGORIES = ['NORMAL', 'PNEUMONIA']

DIR_TRAINING = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/'

DIR_VALIDATION = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/val'

DIR_TEST = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test/'

NEW_SIZE = 100

X_train = []

y_train = []

X_validation = []

y_validation = []

X_test = []

y_test = []



for category in CATEGORIES:

    label = CATEGORIES.index(category)

    path_train = os.path.join(DIR_TRAINING, category)

    path_val = os.path.join(DIR_VALIDATION, category)

    path_test = os.path.join(DIR_TEST, category)

    for img in os.listdir(path_train):

        try:

            img_train = cv2.imread(os.path.join(path_train,img), cv2.IMREAD_COLOR)

            img_train = cv2.resize(img_train, (NEW_SIZE, NEW_SIZE))

            X_train.append(img_train)

            y_train.append(label)

        except Exception as e:

            pass

    for img in os.listdir(path_val):

        try:

            img_val = cv2.imread(os.path.join(path_val,img), cv2.IMREAD_COLOR)

            img_val = cv2.resize(img_val, (NEW_SIZE, NEW_SIZE))

            X_validation.append(img_val)

            y_validation.append(label)

        except Exception as e:

            pass

    for img in os.listdir(path_test):

        try:

            img_test = cv2.imread(os.path.join(path_test,img), cv2.IMREAD_COLOR)

            img_test = cv2.resize(img_test, (NEW_SIZE, NEW_SIZE))

            X_test.append(img_test)

            y_test.append(label)

        except Exception as e:

            pass
X_train = np.array(X_train, dtype="float32").reshape(-1, NEW_SIZE, NEW_SIZE, 3)

y_train = np.asarray(y_train)



X_validation = np.array(X_validation, dtype="float32").reshape(-1, NEW_SIZE, NEW_SIZE, 3)

y_validation = np.asarray(y_validation)



X_test = np.array(X_test, dtype="float32").reshape(-1, NEW_SIZE, NEW_SIZE, 3)

y_test = np.asarray(y_test)
indices_train = np.arange(X_train.shape[0])

np.random.shuffle(indices_train)



X_train = X_train[indices_train]

y_train = y_train[indices_train]
hist_train, bins_train = np.histogram(y_train, bins = [0, 0.5, 1]) 

hist_validation, bins_validation = np.histogram(y_validation, bins = [0, 0.5, 1]) 

hist_test, bins_test = np.histogram(y_test, bins = [0, 0.5, 1]) 



x_labels = ['Train', 'Val', 'Test']

x_hist = np.arange(len(x_labels))

normal = [hist_train[0], hist_validation[0], hist_test[0]]

pneumonia = [hist_train[1], hist_validation[1], hist_test[1]]

width = 0.35

fig, ax = plt.subplots()

rects1 = ax.bar(x_hist - width/2, normal, width, label='Normal')

rects2 = ax.bar(x_hist + width/2, pneumonia, width, label='Pneumonia')

ax.set_xticks(x_hist)

ax.set_xticklabels(x_labels)

ax.legend(['Normal', 'Pneumonia'])

fig.tight_layout()



plt.show()
fig=plt.figure(figsize=(16,16))



for counter, img in enumerate(X_train[:5]):

    ax = fig.add_subplot(1,5,counter+1)

    ax.imshow(X_train[counter,:,:,1], cmap='gray')

    plt.title('Normal')

    ax.axes.get_xaxis().set_visible(False)

    ax.axes.get_yaxis().set_visible(False)



for counter, img in enumerate(X_train[-5:]):

    ax = fig.add_subplot(2,5,counter+1)

    ax.imshow(X_train[-5+counter,:,:,1], cmap='gray')

    plt.title('Pneumonia')

    ax.axes.get_xaxis().set_visible(False)

    ax.axes.get_yaxis().set_visible(False)

    

plt.tight_layout()

plt.show()
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(NEW_SIZE,NEW_SIZE,3), pooling='avg')
model=Sequential()

model.add(base_model)



#model.add(Flatten())

model.add(Dense(256))

model.add(Activation('relu'))

model.add(Dropout(0.2))



#model.add(Dense(128))

#model.add(Activation('relu'))

#model.add(Dropout(0.2))



model.add(Dense(2,activation='softmax'))
base_model.trainable=False
model.compile(loss='categorical_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])
model.summary()
y_cat = to_categorical(y_train)
(X_train2, X_val2, y_train2, y_val2) = train_test_split(X_train, y_cat, test_size=0.1, random_state=42, stratify=y_train)
train_datagen = ImageDataGenerator(

    rescale=1.0/255.0,

    samplewise_center=True,

    rotation_range=20,

    zoom_range=0.15,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.15,

    horizontal_flip=True,

    fill_mode="nearest")



test_datagen = ImageDataGenerator(rescale=1.0/255.0, samplewise_center=True)
batch_size = 64
train_iterator = train_datagen.flow(X_train2, y_train2, batch_size=batch_size, shuffle=False)

test_iterator = test_datagen.flow(X_val2, y_val2, batch_size=batch_size, shuffle=False)
# confirm the scaling works

batchX, batchy = train_iterator.next()

print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
learning_rate_reduction = ReduceLROnPlateau(monitor='accuracy', 

                                            patience=3, 

                                            verbose=1,

                                            factor=0.5, 

                                            min_lr=0.00001)

earlystop = EarlyStopping(patience=10)
earlystop = EarlyStopping(patience=20)



history = model.fit_generator(train_iterator,

                              validation_data=test_iterator,

                              steps_per_epoch=X_train2.shape[0] // batch_size,

                              epochs=100,

                              callbacks=[earlystop, learning_rate_reduction])
predict_datagen = ImageDataGenerator(rescale=1.0/255.0, samplewise_center=True)

predict_iterator = predict_datagen.flow(X_test, y_test, batch_size=len(X_test), shuffle=False)

predicted_label = model.predict(predict_iterator.next())

predicted_label = np.argmax(predicted_label, axis = 1)

print("Model Accuracy on test set: {:.4f}".format(accuracy_score(y_test, predicted_label)))
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper right')

plt.show()
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='lower right')

plt.show()
class_names = ['Normal', 'Pneumonia']

cm  = confusion_matrix(y_test, predicted_label)

plot_confusion_matrix(cm, cmap=plt.cm.Blues, class_names=class_names)
import numpy as np

import pandas as pd

import os

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.layers import Conv2D, Dense, BatchNormalization, Activation, Dropout, MaxPooling2D, Flatten

from keras.optimizers import Adam, RMSprop, SGD

from keras import regularizers

from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau

import datetime

import matplotlib.pyplot as plt

from keras.utils import plot_model
train_dir = '../input/fer2013/train/'

test_dir = '../input/fer2013/test/'



row, col = 48, 48

classes = 7



def count_exp(path, set_):

    dict_ = {}

    for expression in os.listdir(path):

        dir_ = path + expression

        dict_[expression] = len(os.listdir(dir_))

    df = pd.DataFrame(dict_, index=[set_])

    return df

train_count = count_exp(train_dir, 'train')

test_count = count_exp(test_dir, 'test')

print(train_count)

print(test_count)
train_count.transpose().plot(kind='bar')
test_count.transpose().plot(kind='bar')
plt.figure(figsize=(14,22))

i = 1

for expression in os.listdir(train_dir):

    img = load_img((train_dir + expression +'/'+ os.listdir(train_dir + expression)[1]))

    plt.subplot(1,7,i)

    plt.imshow(img)

    plt.title(expression)

    plt.axis('off')

    i += 1

plt.show()
train_datagen = ImageDataGenerator(rescale=1./255,

                                   zoom_range=0.3,

                                   horizontal_flip=True)



training_set = train_datagen.flow_from_directory(train_dir,

                                                batch_size=64,

                                                target_size=(48,48),

                                                shuffle=True,

                                                color_mode='grayscale',

                                                class_mode='categorical')



test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(test_dir,

                                                batch_size=64,

                                                target_size=(48,48),

                                                shuffle=True,

                                                color_mode='grayscale',

                                                class_mode='categorical')
training_set.class_indices
def get_model(input_size, classes=7):

     #Initialising the CNN

    model = tf.keras.models.Sequential()   



    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape =input_size))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(2, 2))

    model.add(Dropout(0.25))



    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))

    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))



    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))

    model.add(Dropout(0.5))

    

    model.add(Dense(classes, activation='softmax'))



    #Compliling the model

    model.compile(optimizer=Adam(lr=0.0001, decay=1e-6), 

                  loss='categorical_crossentropy', 

                  metrics=['accuracy'])

    return model
fernet = get_model((row,col,1), classes)

fernet.summary()
plot_model(fernet, to_file='fernet.png', show_shapes=True, show_layer_names=True)
chk_path = 'ferNet.h5'

log_dir = "checkpoint/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")



checkpoint = ModelCheckpoint(filepath=chk_path,

                             save_best_only=True,

                             verbose=1,

                             mode='min',

                             moniter='val_loss')



earlystop = EarlyStopping(monitor='val_loss', 

                          min_delta=0, 

                          patience=3, 

                          verbose=1, 

                          restore_best_weights=True)

                        

reduce_lr = ReduceLROnPlateau(monitor='val_loss', 

                              factor=0.2, 

                              patience=6, 

                              verbose=1, 

                              min_delta=0.0001)





tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

csv_logger = CSVLogger('training.log')



callbacks = [checkpoint, reduce_lr, csv_logger]
steps_per_epoch = training_set.n // training_set.batch_size

validation_steps = test_set.n // test_set.batch_size



hist = fernet.fit(x=training_set,

                 validation_data=test_set,

                 epochs=60,

                 callbacks=callbacks,

                 steps_per_epoch=steps_per_epoch,

                 validation_steps=validation_steps)
plt.figure(figsize=(14,5))

plt.subplot(1,2,2)

plt.plot(hist.history['accuracy'])

plt.plot(hist.history['val_accuracy'])

plt.title('Model Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend(['train', 'test'], loc='upper left')



plt.subplot(1,2,1)

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title('model Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
train_loss, train_accu = fernet.evaluate(training_set)

test_loss, test_accu = fernet.evaluate(test_set)

print("final train accuracy = {:.2f} , validation accuracy = {:.2f}".format(train_accu*100, test_accu*100))
fernet.save_weights('fernet_bestweight.h5')
y_pred = fernet.predict(training_set)

y_pred = np.argmax(y_pred, axis=1)

class_labels = test_set.class_indices

class_labels = {v:k for k,v in class_labels.items()}



from sklearn.metrics import classification_report, confusion_matrix

cm_train = confusion_matrix(training_set.classes, y_pred)

print('Confusion Matrix')

print(cm_train)

print('Classification Report')

target_names = list(class_labels.values())

print(classification_report(training_set.classes, y_pred, target_names=target_names))



plt.figure(figsize=(8,8))

plt.imshow(cm_train, interpolation='nearest')

plt.colorbar()

tick_mark = np.arange(len(target_names))

_ = plt.xticks(tick_mark, target_names, rotation=90)

_ = plt.yticks(tick_mark, target_names)
y_pred = fernet.predict(test_set)

y_pred = np.argmax(y_pred, axis=1)

class_labels = test_set.class_indices

class_labels = {v:k for k,v in class_labels.items()}



#from sklearn.metrics import classification_report, confusion_matrix

cm_test = confusion_matrix(test_set.classes, y_pred)

print('Confusion Matrix')

print(cm_test)

print('Classification Report')

target_names = list(class_labels.values())

print(classification_report(test_set.classes, y_pred, target_names=target_names))



plt.figure(figsize=(8,8))

plt.imshow(cm_test, interpolation='nearest')

plt.colorbar()

tick_mark = np.arange(len(target_names))

_ = plt.xticks(tick_mark, target_names, rotation=90)

_ = plt.yticks(tick_mark, target_names)
import os

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import tensorflow as tf

print("Tensorflow version " + tf.__version__)



from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, SeparableConv2D, BatchNormalization, MaxPool2D, Dropout

from tensorflow.keras.models import Sequential



from tensorflow.keras.preprocessing.image import ImageDataGenerator
os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray')
base_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray'



train_dir = os.path.join(base_dir, 'train')

test_dir = os.path.join(base_dir, 'val')

validation_dir = os.path.join(base_dir, 'test')



train_normal_dir = os.path.join(train_dir, 'NORMAL')

train_pneumonia_dir = os.path.join(train_dir, 'PNEUMONIA')



validation_normal_dir = os.path.join(validation_dir, 'NORMAL')

validation_pneumonia_dir = os.path.join(validation_dir, 'PNEUMONIA')



test_normal_dir = os.path.join(test_dir, 'NORMAL')

test_pneumonia_dir = os.path.join(test_dir, 'PNEUMONIA')
train_normal_fnames = os.listdir(train_normal_dir)

train_pneumonia_fnames = os.listdir(train_pneumonia_dir)



print(train_normal_fnames[:5])

print(train_pneumonia_fnames[:5])
print('total training normal images :', len(os.listdir(train_normal_dir) ))

print('total training pneumonia images :', len(os.listdir(train_pneumonia_dir) ))



print('total validation normal images :', len(os.listdir(validation_normal_dir) ))

print('total validation pneumonia images :', len(os.listdir(validation_pneumonia_dir) ))



print('total test normal images :', len(os.listdir(test_normal_dir) ))

print('total test pneumonia images :', len(os.listdir(test_pneumonia_dir) ))
base_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/'



fig, ax = plt.subplots(2, 3, figsize=(15, 7))

ax = ax.ravel()

plt.tight_layout()



for i, _set in enumerate(['train', 'val', 'test']):

    set_path = base_dir+_set

    ax[i].imshow(plt.imread(set_path+'/NORMAL/'+os.listdir(set_path+'/NORMAL')[0]), cmap='gray')

    ax[i].set_title('Set: {}, Condition: Normal'.format(_set))

    ax[i+3].imshow(plt.imread(set_path+'/PNEUMONIA/'+os.listdir(set_path+'/PNEUMONIA')[0]), cmap='gray')

    ax[i+3].set_title('Set: {}, Condition: Pneumonia'.format(_set))
train_normal_count = len(os.listdir(train_normal_dir))

train_pneumonia_count = len(os.listdir(train_pneumonia_dir))

total_train_count = train_normal_count + train_pneumonia_count



val_normal_count = len(os.listdir(validation_normal_dir))

val_pneumonia_count = len(os.listdir(validation_pneumonia_dir))

total_val_count = val_normal_count + val_pneumonia_count
weight_for_normal_0 = (1 / train_normal_count)*(total_train_count)/2.0 

weight_for_pneumonia_1 = (1 / train_pneumonia_count)*(total_train_count)/2.0



class_weight = {0: weight_for_normal_0, 1: weight_for_pneumonia_1}



print(f'Weight for class 0: {class_weight[0]}')

print(f'Weight for class 1: {class_weight[1]}')
image_shape = [150, 150]

batch_size = 64

epochs = 30
model = tf.keras.models.Sequential([

    

    # The first convolution

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3),padding = 'same'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D((2, 2), padding = 'same'),

    

    # The second convolution    

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding = 'same'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D((2, 2), padding = 'same'),



    # The third convolution

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding = 'same'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D((2, 2), padding = 'same'),



    # The fourth convolution

    tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding = 'same'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D((2, 2), padding = 'same'),



    # The fifth convolution

    tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding = 'same'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D((2, 2), padding = 'same'),





    # Flatten the results to feed into a DNN

    tf.keras.layers.Flatten(),



    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dropout(0.2),



    tf.keras.layers.Dense(1, activation='sigmoid')

])
model.summary()
# All images will be rescaled by 1./255.

train_datagen = ImageDataGenerator(rescale = 1.0/255,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   zoom_range=0.3,

                                  )



validation_datagen  = ImageDataGenerator(rescale = 1.0/255)



# Flow training images in batches using train_datagen generator

train_generator = train_datagen.flow_from_directory(train_dir,

                                                    batch_size=batch_size,

                                                    class_mode='binary',

                                                    target_size=(image_shape[0], image_shape[1]),

                                                    shuffle=True

                                                   )     



# Flow validation images in batches using test_datagen generator

validation_generator =  validation_datagen.flow_from_directory(validation_dir,

                                                               batch_size=batch_size,

                                                               class_mode  = 'binary',

                                                               target_size = (image_shape[0], image_shape[1]), 

                                                               shuffle=True

                                                              )
metrics = ['accuracy',

           tf.keras.metrics.Precision(name='precision'),

           tf.keras.metrics.Recall(name='recall')

          ]



model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),

              loss='binary_crossentropy',

              metrics = metrics

                 )



learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_recall',

                                                               patience = 3,

                                                               verbose=1,

                                                               factor=0.2,

                                                               min_lr=0.000001)



early_stop = tf.keras.callbacks.EarlyStopping(patience=10,

                                              restore_best_weights=True,

                                              verbose=1)
history = model.fit(train_generator,

                        validation_data=validation_generator,

                        steps_per_epoch=total_train_count//batch_size,

                        epochs=epochs,

                        validation_steps=total_val_count//batch_size,

                        class_weight=class_weight,

                        verbose=2,

                        callbacks=[learning_rate_reduction, early_stop]

                       )
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

precision = history.history['precision']

val_precision = history.history['val_precision']

recall = history.history['recall']

val_recall = history.history['val_recall']





epochs = range(len(acc))

plt.figure()



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, precision, 'r', label='Training precision')

plt.plot(epochs, val_precision, 'b', label='Validation precision')

plt.title('Training and validation precision')

plt.legend()



plt.figure()



plt.plot(epochs, recall, 'r', label='Training recall')

plt.plot(epochs, val_recall, 'b', label='Validation recall')

plt.title('Training and validation recall')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'r', label='Training Loss')

plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
# To get the data and labels to make prediction on the test set which contains 16 files which the model has not seen before

import cv2

test_data = []

true_test_labels = []



for item in ['/NORMAL/', '/PNEUMONIA/']:

    for img in (os.listdir(base_dir + 'val' + item)):

        img = plt.imread(base_dir+'val'+item+img)

        img = cv2.resize(img, (image_shape[0], image_shape[1]))

        img = np.dstack([img, img, img])

        img = img.astype('float32') / 255.

        if item=='/NORMAL/':

            label = 0

        elif item=='/PNEUMONIA/':

            label = 1

        test_data.append(img)

        true_test_labels.append(label)

        

test_data = np.array(test_data)

true_test_labels = np.array(true_test_labels)
prediction = model.predict(test_data)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

tn, fp, fn, tp = confusion_matrix(true_test_labels, np.round(prediction)).ravel()

                                  

precision = tp/(tp+fp)*100

recall = tp/(tp+fn)*100

print(f'Accuracy: {accuracy_score(true_test_labels, np.round(prediction))*100} %')

print(f'Precision: {precision} %')

print(f'Recall: {recall} %')
print(classification_report(true_test_labels, np.round(prediction)))
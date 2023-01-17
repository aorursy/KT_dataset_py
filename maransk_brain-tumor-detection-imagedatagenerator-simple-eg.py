import numpy as np 

from tqdm import tqdm

import cv2

import os

# import shutil

import itertools

# import imutils

import matplotlib.pyplot as plt

# from sklearn.preprocessing import LabelBinarizer

# from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix



# import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

# from plotly import tools



from keras.preprocessing.image import ImageDataGenerator

from keras.applications.vgg16 import VGG16, preprocess_input

from keras import layers

from keras.models import Model, Sequential

from keras.optimizers import Adam, RMSprop

# from keras.callbacks import EarlyStopping



init_notebook_mode(connected=True)

RANDOM_SEED = 123
# os.listdir('../input/train-valtest/all_files/VAL_CROP/'
def target_label(dir_path):

    y = []

    i = 0

    labels = dict()

    for path in tqdm(sorted(os.listdir(dir_path))):

        if not path.startswith('.'):

            labels[i] = path

            if path=='YES':

#                 print(path,len(os.listdir(dir_path + path)))

                yestarget=list(np.ones(len(os.listdir(dir_path + path))))

            if path=='NO':

#                 print(path,len(os.listdir(dir_path + path)))

                notarget=list(np.zeros(len(os.listdir(dir_path + path))))

    y = notarget+yestarget

    print(('{} images are in {} directory').format(len(y),dir_path))

    return y, labels
TRAIN_DIR = '../input/train-valtest/all_files/TRAIN_CROP/'

TEST_DIR = '../input/train-valtest/all_files/TEST_CROP/'

VAL_DIR = '../input/train-valtest/all_files/VAL_CROP/'

# IMG_SIZE = (224,224)

IMG_SIZE = (128,128)

y_train, labels = target_label(TRAIN_DIR)

y_test, _ = target_label(TEST_DIR)

y_val, _ = target_label(VAL_DIR)
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.figure(figsize = (6,6))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=90)

    plt.yticks(tick_marks, classes)

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    cm = np.round(cm,2)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()
# initialize  batch size

BS = 32

train_datagen = ImageDataGenerator()

valid_datagen = ImageDataGenerator(

#     preprocessing_function=preprocess_input

)

train_generator = train_datagen.flow_from_directory(

    '../input/train-valtest/all_files/TRAIN_CROP/',

    target_size=(128, 128),

    batch_size=BS,

    class_mode='binary',

    seed=RANDOM_SEED

)

valid_generator = valid_datagen.flow_from_directory(

    '../input/train-valtest/all_files/VAL_CROP/',

    target_size=(128, 128),

    batch_size=25,

    class_mode='binary',

    seed=RANDOM_SEED

)
# os.listdir('')
IMG_SIZE = (128,128)

# load base model

vgg16_weight_path = '../input/vgg16pretrained/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

base_model = VGG16(

    weights=vgg16_weight_path,

    include_top=False, 

    input_shape=IMG_SIZE + (3,)

)
# model = Sequential()



# # Must define the input shape in the first layer of the neural network

# model.add(layers.Conv2D(filters=16,kernel_size=9, padding='same', activation='relu', input_shape=(128,128,3))) 

# model.add(layers.MaxPooling2D(pool_size=2))

# model.add(layers.Dropout(0.45))



# model.add(layers.Conv2D(filters=16,kernel_size=9,padding='same', activation='relu'))

# model.add(layers.MaxPooling2D(pool_size=2))

# model.add(layers.Dropout(0.25))



# model.add(layers.Conv2D(filters=36, kernel_size=9, padding='same', activation='relu'))

# model.add(layers.MaxPooling2D(pool_size=2))

# model.add(layers.Dropout(0.25))



# model.add(layers.Flatten())



# model.add(layers.Dense(512, activation='relu'))

# model.add(layers.Dropout(0.15))





# model.add(layers.Dense(1, activation='sigmoid'))



# # Take a look at the model summary

# model.summary()
NUM_CLASSES = 1



model = Sequential()

model.add(base_model)

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(NUM_CLASSES, activation='sigmoid'))



model.layers[0].trainable = False

model.summary()
# len(X_train)//BS
NUM_EPOCHS = 10

model.compile(loss='binary_crossentropy',

             optimizer='Adam',

             metrics=['accuracy'])

history = model.fit_generator(

    train_generator,

    steps_per_epoch=len(y_train)//25,

#     steps_per_epoch=70,

    epochs=NUM_EPOCHS,

    validation_data = valid_generator, 

    validation_steps=len(y_val)//25,  

#     validation_steps=10,

)
# plot model performance

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs_range = range(1, len(history.epoch) + 1)



plt.figure(figsize=(15,5))



plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Train Set')

plt.plot(epochs_range, val_acc, label='Val Set')

plt.legend(loc="best")

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.title('Model Accuracy')



plt.subplot(1, 2, 2)

plt.plot(epochs_range[1:], loss[1:], label='Train Set')

plt.plot(epochs_range, val_loss, label='Val Set')

plt.legend(loc="best")

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.title('Model Loss')



plt.tight_layout()

plt.show()
test_generator = valid_datagen.flow_from_directory(

    '../input/train-valtest/all_files/TEST_CROP/',

    target_size=(128, 128),

    batch_size=5,

    class_mode='binary',

    shuffle=False

)

test_generator_val = valid_datagen.flow_from_directory(

    '../input/train-valtest/all_files/VAL_CROP/',

    target_size=(128, 128),

    batch_size=10,

    class_mode='binary',

    shuffle=False

)

test_generator_train = valid_datagen.flow_from_directory(

    '../input/train-valtest/all_files/TRAIN_CROP/',

    target_size=(128, 128),

    batch_size=50,

    class_mode='binary',

    shuffle=False

)
predIdx=model.predict_generator(test_generator,

	steps= 2)

predIdx=[1 if x>0.5 else 0 for x in predIdx]



accuracy = accuracy_score(y_test, np.round(predIdx))

print('Test Accuracy = %.2f' % accuracy)



confusion_mtx = confusion_matrix(y_test, np.round(predIdx)) 

cm = plot_confusion_matrix(confusion_mtx, classes = list(labels.items()), normalize=False)
# ind_list = np.argwhere((y_test == predIdx) == False)[:, -1]

# if ind_list.size == 0:

#     print('There are no missclassified images.')

# else:

#     for i in ind_list:

#         plt.figure()

#         plt.imshow(X_test[i])

#         plt.xticks([])

#         plt.yticks([])

#         plt.title(('Actual class: {}\nPredicted class: {}').format(y_test[i],predIdx[i]))

#         plt.show()
predIdx=model.predict_generator(test_generator_val,

	steps= 5)

predIdx=[1 if x>0.5 else 0 for x in predIdx]

len(predIdx)
accuracy = accuracy_score(y_val, np.round(predIdx))

print('VAL Accuracy = %.2f' % accuracy)



confusion_mtx = confusion_matrix(y_val, np.round(predIdx)) 

cm = plot_confusion_matrix(confusion_mtx, classes = list(labels.items()), normalize=False)
### you can play with batch size, steps to limit how many images to use for prediction

predIdx=model.predict_generator(test_generator_train,

	steps= 4)

predIdx=[1 if x>0.5 else 0 for x in predIdx]

len(predIdx)
accuracy = accuracy_score(y_train, np.round(predIdx))

print('TRAIN Accuracy = %.2f' % accuracy)



confusion_mtx = confusion_matrix(y_train, np.round(predIdx)) 

cm = plot_confusion_matrix(confusion_mtx, classes = list(labels.items()), normalize=False)
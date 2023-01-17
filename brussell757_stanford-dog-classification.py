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
images_dir = os.path.abspath('../input/stanford-dogs-dataset/images/Images')

annotations_dir = os.path.abspath('../input/stanford-dogs-dataset/annotations/Annotation')

vgg_16_weights = os.path.abspath('../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

densenet_weights = os.path.abspath('../input/densenet-keras/DenseNet-BC-121-32-no-top.h5')

xception_weights = os.path.abspath('../input/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5')
WIDTH = 72

HEIGHT = 72

CHANNELS = 3

LEARNING_RATE = 1e-4

EPOCHS = 500

BATCH_SIZE = 64

NUM_CLASSES = 0
from glob import glob



def get_images_labels(directory):

    labels_images = {}

    for _dir in glob(directory + '/*'):

        images = []

        curr_label = os.path.basename(_dir)

        for _file in glob(directory + '/' + curr_label + '/*.jpg'):

            images.append(_file)

            

        labels_images.setdefault(curr_label, images)

        

    return labels_images



labels_images = get_images_labels(images_dir)
NUM_CLASSES = len(labels_images)
from PIL import Image



def preprocess_images(dictionary, width = WIDTH, height = HEIGHT):

    images = []

    labels = []

    for k,v in dictionary.items():

        for path in v:

            img = Image.open(path)

            img = img.convert('RGB')

            img = img.resize((height, width))

            img = np.asarray(img)

            img = img / 255.

            

            images.append(img)

            labels.append(str(k.split('-')[1].lower()))

    

    return np.array(images), np.array(labels)



images, labels = preprocess_images(labels_images)

    

print('images shape:',images.shape)

print('labels shape:',labels.shape)
import matplotlib.pyplot as plt



def display_images(images, labels, num_images = 25):

    plt.figure(figsize = (25,25))

    for i in range(num_images):

        index = np.random.randint(0,images.shape[0])

        plt.subplot(5,5,i+1)

        plt.imshow(images[index])

        plt.title('Label: {}'.format(str(labels[index])))

        

    plt.show()

    

display_images(images, labels)
from sklearn.preprocessing import LabelEncoder

from keras.utils import to_categorical



encoder = LabelEncoder()

labels_encoded = encoder.fit_transform(labels)

labels_encoded = to_categorical(labels_encoded, NUM_CLASSES)



print('labels shape:',labels_encoded.shape)
import keras



#vgg16_base = keras.applications.vgg16.VGG16(include_top = False, weights = vgg_16_weights, input_shape = (HEIGHT, WIDTH, CHANNELS))

#densenet_base = keras.applications.densenet.DenseNet121(include_top = False, weights = densenet_weights, input_shape = (HEIGHT, WIDTH, CHANNELS))

xception_base = keras.applications.xception.Xception(include_top = False, weights = xception_weights, input_shape = (HEIGHT, WIDTH, CHANNELS))



def build_model():

    

    backbone = xception_base

    backbone_output = backbone.output

  

    X = keras.layers.GlobalAveragePooling2D()(backbone_output)

    

    X = keras.layers.Dense(1024)(X)

    X = keras.layers.Activation('relu')(X)

    X = keras.layers.Dropout(0.5)(X)

    

    X = keras.layers.Dense(1024)(X)

    X = keras.layers.Activation('relu')(X)

    X = keras.layers.Dropout(0.5)(X)

    

    X = keras.layers.Dense(512)(X)

    X = keras.layers.Activation('relu')(X)

    X = keras.layers.Dropout(0.5)(X)

    

    X = keras.layers.Dense(NUM_CLASSES)(X)

    outputs = keras.layers.Activation('softmax')(X)

    

    model = keras.models.Model(inputs = backbone.input, outputs = outputs)



    return model

model = build_model()



model.compile(optimizer = keras.optimizers.Adam(lr = LEARNING_RATE), loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.summary()
from sklearn.model_selection import train_test_split

X_train, X_dev, Y_train, Y_dev = train_test_split(images, labels_encoded, test_size = 0.2, random_state = 1029)

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size = 0.35, random_state = 1029)



print('X_train shape:', X_train.shape)

print('Y_train shape:', Y_train.shape)

print('X_dev shape:', X_dev.shape)

print('Y_dev shape:', Y_dev.shape)

print('X_test shape:', X_test.shape)

print('Y_test shape:', Y_test.shape)
from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rotation_range = 45,

                                   width_shift_range = 0.2,

                                   height_shift_range = 0.2,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



dev_datagen = ImageDataGenerator(rotation_range = 45,

                                 width_shift_range = 0.2,

                                 height_shift_range = 0.2,

                                 shear_range = 0.2,

                                 zoom_range = 0.2,

                                 horizontal_flip = True)



train_gen = train_datagen.flow(X_train, Y_train, batch_size = BATCH_SIZE)



dev_gen = dev_datagen.flow(X_dev, Y_dev, batch_size = BATCH_SIZE)



train_steps = train_gen.n // BATCH_SIZE

dev_steps = dev_gen.n // BATCH_SIZE
log_dir = 'logs'

if not os.path.isdir(log_dir): os.mkdir(log_dir)

    

model_checkpoint_dir = 'checkpoints'

if not os.path.isdir(model_checkpoint_dir): os.mkdir(model_checkpoint_dir)
callbacks = [keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 3, verbose = 1),

             keras.callbacks.TensorBoard(log_dir = './logs', histogram_freq = 0, batch_size = BATCH_SIZE, write_graph = True, write_grads = True, write_images = False),

             keras.callbacks.ModelCheckpoint(filepath = model_checkpoint_dir + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor = 'val_loss', save_best_only = True, verbose = 1, period = 25)]
history = model.fit_generator(train_gen,

                              steps_per_epoch = train_steps,

                              epochs = EPOCHS,

                              validation_data = dev_gen,

                              validation_steps = dev_steps,

                              callbacks = callbacks,

                              verbose = 2)
def display_metrics_graph(history):

    acc, val_acc, loss, val_loss = history.history['acc'], history.history['val_acc'], history.history['loss'], history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    

    plt.figure(figsize = (10,7))

    

    plt.subplot(1,2,1)

    plt.plot(epochs, acc, c = 'b', label = 'training accuracy')

    plt.plot(epochs, val_acc, c = 'g', label = 'validation accuracy')

    plt.title('Training vs. Validation Accuracy')

    plt.xlabel('epoch')

    plt.ylabel('accuracy')

    plt.legend()

    

    plt.subplot(1,2,2)

    plt.plot(epochs, loss, c = 'b', label = 'training loss')

    plt.plot(epochs, val_loss, c = 'g', label = 'validation loss')

    plt.title('Training vs. Validation Loss')

    plt.xlabel('epoch')

    plt.ylabel('loss')

    plt.legend()

    

    plt.show()

        

display_metrics_graph(history)
eval_loss, eval_acc = model.evaluate(X_test, Y_test)

print('Evaluation Loss: {}, Evaluation Accuracy: {}'.format(eval_loss, eval_acc * 100))
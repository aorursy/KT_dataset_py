import os

import shutil

import glob

import urllib.request



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



import tensorflow as tf 

glob.glob('../input/chest-xray-pneumonia/chest_xray/chest_xray/*/*')

img_normal = plt.imread('../input/chest-xray-pneumonia/chest_xray/chest_xray/train/NORMAL/IM-0131-0001.jpeg')

img_penumonia_bacteria = plt.imread('../input/chest-xray-pneumonia/chest_xray/chest_xray/train/PNEUMONIA/person1017_bacteria_2948.jpeg')

img_penumonia_virus = plt.imread('../input/chest-xray-pneumonia/chest_xray/chest_xray/train/PNEUMONIA/person1021_virus_1711.jpeg')



plt.figure(figsize=(12, 5))



plt.subplot(1,3,1).set_title('NORMAL')

plt.imshow(img_normal, cmap='gray')



plt.subplot(1,3,2).set_title('PNEUMONIA')

plt.imshow(img_penumonia_bacteria, cmap='gray')



plt.subplot(1,3,3).set_title('PNEUMONIA')

plt.imshow(img_penumonia_virus, cmap='gray')



plt.tight_layout()
def get_labeled_files(folder):

    x = []

    y = []

    

    for folderName in os.listdir(folder):

        if not folderName.startswith('.'):

            if folderName in ['NORMAL']:

                label = 0

            elif folderName in ['PNEUMONIA']:

                label = 1

            else:

                label = 2

                continue # we do not investigate other dirs

            for image_filename in os.listdir(folder + folderName):

                img_path = folder + folderName + '/' + image_filename

                if img_path is not None and str.endswith(img_path, 'jpeg'):

                    x.append(img_path)

                    y.append(label)

    

    x = np.asarray(x)

    y = np.asarray(y)

    return x, y
x, y = get_labeled_files('../input/chest-xray-pneumonia/chest_xray/chest_xray/train/')



list(zip(x, y))[:10]
#refer: https://www.tensorflow.org/api_docs/python/tf/



NUM_CLASSES = 2



# This function takes image paths as arguments and reads corresponding images

def input_parser(img_path, label):

    # convert the label to one-hot encoding

    one_hot = tf.one_hot(label, NUM_CLASSES)

    # read the img from file and decode it using tf

    img_file = tf.read_file(img_path)

    img_decoded = tf.image.decode_jpeg(img_file, channels=3, name="decoded_images")

    return img_decoded, one_hot



# This function takes image and resizes it to smaller format (150x150)

def image_resize(images, labels):

    # Be very careful with resizing images like this and make sure to read the doc!

    # Otherwise, bad things can happen - https://hackernoon.com/how-tensorflows-tf-image-resize-stole-60-days-of-my-life-aba5eb093f35

    resized_image = tf.image.resize_images(images, (150, 150), align_corners=True)

    resized_image_asint = tf.cast(resized_image, tf.int32)

    return resized_image_asint, labels    

#refer:https://kratzert.github.io/2017/06/15/example-of-tensorflows-new-input-pipeline.html



# Since it uses lazy evaluation, the images will not be read after calling build_pipeline_plan()

# We need to use iterator defined here in tf context

def build_pipeline_plan(img_paths, labels, batch_size):



    # We build a tensor of image paths and labels

    tr_data = tf.data.Dataset.from_tensor_slices((img_paths, labels))

    # First step of input pipeline - read images in paths as jpegs

    tr_data_imgs = tr_data.map(input_parser)

    # Apply resize to each image in the pipeline

    tr_data_imgs = tr_data_imgs.map(image_resize)

    # Gives us opportuinty to batch images into small groups

    tr_dataset = tr_data_imgs.batch(batch_size)

    # create TensorFlow Iterator object directly from input pipeline

    iterator = tr_dataset.make_one_shot_iterator()

    next_element = iterator.get_next()

    return next_element



# Function to execute defined pipeline in Tensorflow session

def process_pipeline(next_element):

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        # get each element of the training dataset until the end is reached

        # in our case only one iteration since we read everything as 1 batch

        # can be multiple iterations if we decrease BATCH_SIZE to eg. 10

        images = []

        labels_hot = []

        while True:

            try:

                elem = sess.run(next_element)

                images = elem[0]

                labels_hot = elem[1]

            except tf.errors.OutOfRangeError:

                print("Finished reading the dataset")

                return images, labels_hot

def load_dataset(path, batch_size):

    tf.reset_default_graph()

    files, labels = get_labeled_files(path)

    p = tf.constant(files, name="train_imgs")

    l = tf.constant(labels, name="train_labels")

    

    next_element = build_pipeline_plan(p, l, batch_size=batch_size)

    imgs, labels = process_pipeline(next_element)

    return imgs, labels
x_train, y_train = load_dataset("../input/chest-xray-pneumonia/chest_xray/chest_xray/train/", 6000)

x_test, y_test = load_dataset("../input/chest-xray-pneumonia/chest_xray/chest_xray/test/", 6000)

x_val, y_val = load_dataset("../input/chest-xray-pneumonia/chest_xray/chest_xray/val/", 6000)

print("Training Dataset")

print(x_train.shape)

print(y_train.shape)

print("\nTesting Dataset")

print(x_test.shape)

print(y_test.shape)

print("\n Validation  Dataset")

print(x_val.shape)

print(y_val.shape)
print(x_val)
import matplotlib.pyplot as plt

import seaborn as sns 



plt.subplot(1,3,1)

sns.countplot(np.argmax(y_train, axis=1)).set_title('TRAIN')



plt.subplot(1,3,2)

sns.countplot(np.argmax(y_test, axis=1)).set_title('TEST')



plt.subplot(1,3,3)

sns.countplot(np.argmax(y_val, axis=1)).set_title('VALIDATION')



plt.tight_layout()

print(x_train.shape)



plt.figure(figsize=(5, 3))



y_train_classes = np.argmax(y_train, axis = 1)



plt.subplot(1,2,1).set_title('NORMAL')

plt.imshow(x_train[np.argmax(y_train_classes == 0)])



plt.subplot(1,2,2).set_title('PNEUMONIA')

plt.imshow(x_train[np.argmax(y_train_classes == 1)])



plt.tight_layout()

import keras

from keras import backend as K

from keras.models import Model

from keras.layers import Flatten, Dense, BatchNormalization, Dropout

from keras.applications.vgg16 import VGG16

from keras.utils import plot_model

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

#from keras.applications.vgg19 import VGG19

#from keras.applications.densenet import DenseNet

from keras.applications.inception_v3 import InceptionV3



K.clear_session()



NUM_CLASSES = 2



base_model = VGG16(weights='../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(150, 150, 3))

#base_model = VGG19(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

#base_model = DenseNet(blocks=10, weights='imagenet', include_top=False, input_shape=(150, 150, 3))

#base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(150, 150, 3))



x = base_model.output

x = Flatten()(x)

x = Dense(NUM_CLASSES, activation='softmax')(x)



model = Model(inputs=base_model.input, outputs=x)

base_model.summary()

model.summary()

type(model)



plot_model(model, to_file='model.png')

SVG(model_to_dot(model).create(prog='dot', format='svg'))
#function to print the model layers



def print_layers(model):

    for idx, layer in enumerate(model.layers):

        print("layer {}: {}, trainable: {}".format(idx, layer.name, layer.trainable))

#setting VGG16 layers to False, not to be trained



for layer in model.layers[0:18]:

    layer.trainable = False

    

print_layers(model)
model.trainable_weights
import numpy as np

import keras.backend as K

from itertools import product

from functools import partial
#credit :https://github.com/keras-team/keras/issues/9598



def w_categorical_crossentropy(y_true, y_pred, weights):

    nb_cl = len(weights)

    final_mask = K.zeros_like(y_pred[:, 0])

    y_pred_max = K.max(y_pred, axis=1)

    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))

    y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max), K.floatx())

    for c_p, c_t in product(range(nb_cl), range(nb_cl)):

        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])

    return K.categorical_crossentropy(y_pred, y_true) * final_mask





w_array = np.ones((2,2))

w_array[1,0] = 30 # penalizing FN

w_array[0,1] = 1 # penalizing FP



spec_loss = lambda y_true, y_pred: w_categorical_crossentropy(y_true, y_pred, weights=w_array)

#setting the hyper parameters of the model

# Create the loss function object using the wrapper function above

optimizer = keras.optimizers.Adam(lr=0.0001)



model.compile(loss='categorical_crossentropy',     

              optimizer=optimizer, 

              metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping



# refer: https://stackoverflow.com/questions/42112260/how-do-i-use-the-tensorboard-callback-of-keras

        

# This callback writes logs for TensorBoard to save the training

tensorboard = TensorBoard(

    log_dir='./Graph', 

    histogram_freq=0,  

    write_graph=True

)



from sklearn.utils import class_weight

y_labels = np.argmax(y_train, axis=1)

classweight = class_weight.compute_class_weight('balanced', np.unique(y_labels), y_labels)

print(classweight)
# prepare a directory to store the model weights

os.makedirs('./model', exist_ok=True)



history = model.fit(

    x=x_train, y=y_train,

    class_weight=classweight,

    validation_split=0.3,

    callbacks=[tensorboard],

    shuffle=True,

    batch_size=256,

    epochs=60,

    verbose=1

)



def plot_learning_curves(history):

    plt.figure(figsize=(12,4))

    

    plt.subplot(1,2,1)

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'val'], loc='upper left')

    

    plt.subplot(1,2,2)

    plt.plot(history.history['acc'])

    plt.plot(history.history['val_acc'])

    plt.title('model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['train', 'val'], loc='upper left')

    

    plt.tight_layout()

    

plot_learning_curves(history)
score = model.evaluate(x_test, y_test, verbose=0)

print('Model Loss: {}, Accuracy: {}'.format(score[0], score[1]))
from sklearn.metrics import confusion_matrix

from mlxtend.plotting import plot_confusion_matrix



y_pred = model.predict(x_test)

# to get the prediction, we pick the class with with the highest probability

y_pred_classes = np.argmax(y_pred, axis = 1) 

y_true = np.argmax(y_test, axis = 1) 





conf_mtx = confusion_matrix(y_true, y_pred_classes) 

plot_confusion_matrix(conf_mtx, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)

plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)

plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)

plt.show()
#refer:https://www.kaggle.com/c/dauphine-203-machine-learning-competition/discussion/52755



from sklearn.metrics import roc_curve

from sklearn.metrics import auc



fpr = dict()

tpr = dict()

roc_auc = dict()



for i in range(NUM_CLASSES):

    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])

    roc_auc[i] = auc(fpr[i], tpr[i])

                                 

plt.figure(figsize=(7, 5))



for i in range(NUM_CLASSES):

    plt.plot(fpr[i], tpr[i], lw=2,

             label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    

plt.plot(fpr[0], fpr[0], 'k-', label = 'random guessing')



plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.legend(loc="lower right")



plt.tight_layout()


# saving a model

model.save("../working/model.h5")

print("Saved model to disk")
print(os.listdir("../working"))

#reloading a model



from keras.models import load_model



new_model = load_model('../working/model.h5')

    

new_model.summary()
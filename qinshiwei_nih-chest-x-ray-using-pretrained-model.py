import cv2

from keras.preprocessing.image import load_img, img_to_array

import os

import random

import matplotlib.pylab as plt

from glob import glob

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from IPython.display import display
# check package versions 

import pkg_resources

print(pkg_resources.get_distribution("pandas"))

print(pkg_resources.get_distribution("opencv-python"))

print(pkg_resources.get_distribution("scikit-learn"))

print(pkg_resources.get_distribution("keras"))

print(pkg_resources.get_distribution("tensorflow"))
# folder structure

print("\nList of folder: \n")

for dirpath, dirnames, filenames in os.walk('/kaggle/input'):

    for dirname in dirnames:

        print(os.path.join(dirpath, dirname))

        

        

# Image files

print("\nList of images: \n")

for dirpath, dirnames, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirpath, filename))
# ../input/

PATH = os.path.abspath(os.path.join('..', 'input'))



# ../input/sample/images/

SOURCE_IMAGES = os.path.join(PATH, "sample", "images")



# ../input/sample/images/*.png

images = glob(os.path.join(SOURCE_IMAGES, "*.png"))



# Load labels

labels = pd.read_csv('../input/sample_labels.csv')

unique_labels = labels['Finding Labels'].unique()

print("there are {} unique lables, {} include the word 'infiltration'"

      .format(len(unique_labels), 

              np.array(['infiltration' in label.lower() for label in unique_labels]).sum()))

display(labels.head())
img_r = images[0] 



img_tf_original = img_to_array(load_img(path=img_r))/255

img_tf_resized = img_to_array(load_img(path=img_r, target_size = (224, 224)))/255



img_cv_original = cv2.imread(img_r)

img_cv_resized = cv2.resize(img_cv_original, (224, 224), interpolation=cv2.INTER_CUBIC)

img_cv_original = img_cv_original/255

img_cv_resized = img_cv_resized/255



print("max pixel value difference between originals = {}".format(np.abs(img_tf_original - img_cv_original).mean()))

print("max pixel value difference between resized = {}".format(np.abs(img_tf_resized - img_cv_resized).mean()))

                                                               

plt.figure(figsize=(10,10))

plt.subplot(221)

plt.title('tf_original')

display(plt.imshow(img_tf_original))



plt.subplot(222)

plt.title('tf_resized')

display(plt.imshow(img_tf_resized))



plt.subplot(223)

plt.title('cv_original')

display(plt.imshow(img_cv_original))



plt.subplot(224)

plt.title('cv_resized')

display(plt.imshow(img_cv_resized))
disease="infiltration"



x = [] 

y = [] 

WIDTH = 224 

HEIGHT = 224 



for img in images:

    base = os.path.basename(img)  # image file name

    finding = labels.loc[lambda df: df['Image Index'] == base]["Finding Labels"].values[0]



    # Read and resize image

    image_resized =  img_to_array(load_img(path=img, target_size = (WIDTH, HEIGHT)))

    x.append(image_resized)



    # Labels

    if disease in finding.lower():

        y.append(1)

    else:

        y.append(0)

        

x_image_array = np.asarray(x)/255    # normalized by 255        

y_labels = np.asarray(y)

print("Total {} samples, image size {}, {} samples with infiltration"

      .format(x_image_array.shape[0], x_image_array[0].shape, y_labels.sum()))
# train_test split

x_train, x_test, y_train, y_test = train_test_split(x_image_array, y_labels, test_size=0.2)

print("{} samples for training, {} samples for testing".format(x_train.shape[0], x_test.shape[0]))
# The pretrained weights are downloaded from github. For this to work, internet need to be enabled. 



from keras.applications.resnet50 import ResNet50

base_model = ResNet50(weights='imagenet')



# Show model summary and layer names

# base_model.summary()



print("\nThe layer we will replace:\n{}".format(base_model.layers[-1].get_config()))

print("\nthe layer we will use as input to the new layers we add:\n{}".format(base_model.layers[-2].get_config()))
# We will replace the last "fc1000" dense layer with two dense layers and replace the softmax activation with sigmoid 

# because there are only two classes now. 

from keras.models import Model

from keras.layers import Dense



x = base_model.layers[-2].output

x = Dense(512, activation='relu')(x)

predictions = Dense(1, activation='sigmoid')(x)



new_model = Model(inputs=base_model.input, outputs=predictions)



# freeze all layers taken from pretrained ResNet50

for layer in base_model.layers:

    layer.trainable = False



new_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])    

# new_model.summary() # show model summary
# Just to confirm the old layers are NOT trainable 

new_model.get_layer(name='bn5c_branch2c').get_config()['trainable']
%%time



# 10 epochs gave over 80% accuracy

history = new_model.fit(x = x_train, 

                        y = y_train, 

                        epochs = 10, 

                        batch_size = 200, 

                        validation_split=0.2, 

                        verbose=1)



# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
new_model.evaluate(x=x_test, y=y_test, batch_size=100)
import sklearn

import itertools

from sklearn.metrics import confusion_matrix

dict_characters = {0: 'No Infiltration Observed', 1: 'Pulmonary Infiltration Observed'}

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

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

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



a=x_train

b=y_train

c=x_test

d=y_test

y_pred = new_model.predict(c)

y_pred_classes = (y_pred > 0.5).astype(int)

confusion_mtx = confusion_matrix(d, y_pred_classes) 

plot_confusion_matrix(confusion_mtx, classes = list(dict_characters.values()))
# calculate majority class accuracy of the test data

# (accuracy if we simply predict all samples to the majority class - 0 in our case)

majority_class_accuracy = 1 - y_test.sum()/len(y_test)

print("majority_class_accuracy =  {}".format(majority_class_accuracy))
idx_infiltration = np.where(y_train>0)[0]

idx_no_infiltration = np.where(y_train==0)[0]



r_infiltration = np.random.choice(idx_infiltration, 4)

r_no_infiltration = np.random.choice(idx_no_infiltration, 4)



plt.figure(figsize=(16,8))

for i in range(4):

    plt.subplot(2,4,i+1)

    plt.imshow(x_train[r_infiltration[i]])

    plt.title('infiltration')

for i in range(4):

    plt.subplot(2,4,i+5)

    plt.imshow(x_train[r_no_infiltration[i]])

    plt.title('no infiltration')
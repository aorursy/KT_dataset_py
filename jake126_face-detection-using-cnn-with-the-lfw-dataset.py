# general functions

import os

import cv2

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

from collections import OrderedDict

# visualization

from PIL import Image

# identifying faces

%pip install mtcnn

from mtcnn.mtcnn import MTCNN

# visualizing bounding boxes

import matplotlib.patches as patches

# CNN

import keras

from sklearn.model_selection import train_test_split

# Moving files between directories

import shutil

from shutil import unpack_archive

from subprocess import check_output



dataset_path = "../input/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/"
# Data read-in and cleaning



lfw_allnames = pd.read_csv("../input/lfw-dataset/lfw_allnames.csv")

matchpairsDevTest = pd.read_csv("../input/lfw-dataset/matchpairsDevTest.csv")

matchpairsDevTrain = pd.read_csv("../input/lfw-dataset/matchpairsDevTrain.csv")

mismatchpairsDevTest = pd.read_csv("../input/lfw-dataset/mismatchpairsDevTest.csv")

mismatchpairsDevTrain = pd.read_csv("../input/lfw-dataset/mismatchpairsDevTrain.csv")

pairs = pd.read_csv("../input/lfw-dataset/pairs.csv")

# tidy pairs data: 

pairs = pairs.rename(columns ={'name': 'name1', 'Unnamed: 3': 'name2'})

matched_pairs = pairs[pairs["name2"].isnull()].drop("name2",axis=1)

mismatched_pairs = pairs[pairs["name2"].notnull()]

people = pd.read_csv("../input/lfw-dataset/people.csv")

# remove null values

people = people[people.name.notnull()]

peopleDevTest = pd.read_csv("../input/lfw-dataset/peopleDevTest.csv")

peopleDevTrain = pd.read_csv("../input/lfw-dataset/peopleDevTrain.csv")
print("Summary:")

print("There are " + 

      str(lfw_allnames.shape[0]) + 

      " unique celebrities in the entire dataset, of whom " + 

      str(sum(lfw_allnames.images > 1)) + 

      " are represented by multiple images. The entire number of images available is " + 

      str(sum(lfw_allnames.images)) + 

      ". The most represented celebrity is " + 

      str(lfw_allnames.iloc[lfw_allnames['images'].idxmax()][0]) + 

      ", with " + 

      str(max(lfw_allnames.images)) + 

      " unique images in the dataset.")
# shape data frame so there is a row per image, matched to relevant jpg file

image_paths = lfw_allnames.loc[lfw_allnames.index.repeat(lfw_allnames['images'])]

image_paths['image_path'] = 1 + image_paths.groupby('name').cumcount()

image_paths['image_path'] = image_paths.image_path.apply(lambda x: '{0:0>4}'.format(x))

image_paths['image_path'] = image_paths.name + "/" + image_paths.name + "_" + image_paths.image_path + ".jpg"

image_paths = image_paths.drop("images",1)



# take a random sample: 80% of the data for the test set

lfw_train, lfw_test = train_test_split(image_paths, test_size=0.2)

lfw_train = lfw_train.reset_index().drop("index",1)

lfw_test = lfw_test.reset_index().drop("index",1)



# verify that there is a mix of seen and unseen individuals in the test set

print(len(set(lfw_train.name).intersection(set(lfw_test.name))))

print(len(set(lfw_test.name) - set(lfw_train.name)))



# both comprehensively non-empty - we are ok to procede.

# N.B. although we don't use this training/test split in the following model, this is the format of the data we

# would use in applying models to the full dataset
# verify resolution of all images is consistent

widths = []

heights = []

files = image_paths.image_path

for file in files:

    path = "../input/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/" + str(file)

    im = Image.open(path)

    widths.append(im.width)

    heights.append(im.height)



pd.DataFrame({'height':heights,'width':widths}).describe()



# all 250 x 250 resolution
image_paths['name'].value_counts()[:10].plot(kind = "bar")
ind_counts = image_paths.groupby('name').count().image_path

print(str(sum(ind_counts[ind_counts==1])) + " individuals, which is " + str(round(100*(sum(ind_counts[ind_counts==1])/sum(ind_counts)))) + "% of the total individuals considered, are only represented by a single image in this dataset.")
im = Image.open("../input/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/" + str(lfw_train.image_path[0]))

plt.imshow(im)
# Use out-of-the-box MTCNN algorithm to identify facial boundaries on a sample image

detector = MTCNN()

image = cv2.imread("../input/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/" + str(lfw_train.image_path[0]))

result = detector.detect_faces(image)

# That's it - the results have already been generated! Now we extract the boundaries that have been identified 

bounding_box = result[0]['box']

keypoints = result[0]['keypoints']



# Finally, we plot the image along with the face's bounding box and internal features (eyes, nose, and mouth corners)

fig,ax = plt.subplots(1)

ax.imshow(image)

# Create a Rectangle patch

rect = patches.Rectangle(bounding_box[0:2],bounding_box[2],bounding_box[3],linewidth=1,edgecolor='r',facecolor='none')

# Add the patch to the Axes

ax.add_patch(rect)

# Visualize all the internal features

for key in keypoints:

    rect_key = patches.Rectangle(keypoints[key],1,1,linewidth=10,edgecolor='r',facecolor='none')

    ax.add_patch(rect_key)

# Show the plots

plt.show()
not_george = image_paths[image_paths.name!="George_W_Bush"].sample(530)

not_george.name = "not_George"

GWB_data = pd.concat([image_paths[image_paths.name=="George_W_Bush"],

                     not_george])
# withhold final test data

bush_train, bush_test = train_test_split(GWB_data, test_size=0.2)

#  It is also useful to reserve a validation set in the training data, for use in the model construction

bush_train, bush_val = train_test_split(bush_train,test_size=0.2)
# initialize sequential network

from keras.models import Sequential

# include convolutional layers

from keras.layers import Conv2D

# Pooling layers

from keras.layers import MaxPooling2D

# flatten layers into single vector

from keras.layers import Flatten

from keras.layers import Dense
# use sequential classifier - standard CNN implementation with straightforward single input

Bush_classifier = Sequential()

# We then add layers of complexity to this classifier. The first layer we add is a 2D convolutional layer (as 

# explained above); we pass over the image with a 3x3 window and expect 250*250 pixel input.

# The activation function to introduce non-linearity  is 'relu': Rectified Linear Units, a simple activation function 

# with low computational speeds. One potential downside is that ReLU can lead to "dying neurons", where a neuron is 

# attributed zero influence on classification and gets stuck in this state; we can see output where many neurons have 

# zero influence on the model. If this behaviour manifested in further analysis, an alternate activation function e.g. 

# 'leaky ReLU' would be worth exploring as an alternative

Bush_classifier.add(Conv2D(32, (3, 3), input_shape = (250, 250, 3), activation = 'relu'))



# We now add the pooling layer to reduce the dimension

Bush_classifier.add(MaxPooling2D(pool_size = (2, 2)))



# The next step is to flatten the data, reducing the feature maps to a 1D array

Bush_classifier.add(Flatten())



# We then add a fully connected layer - traditional Multi Layer Perceptron

Bush_classifier.add(Dense(units = 128, activation = 'relu'))



# as we are just training on 'Bush' vs 'non-Bush', we only need to add one classification unit

Bush_classifier.add(Dense(units = 1, activation = 'sigmoid'))



# We are now ready to compile the model. It is possible, and advisable, to introduce more layers to the network, but

# as this model is exploratory we are keeping things straightforward for now (this would change in future iterations)

Bush_classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# define a custom function to move images to a new train/test/val directory



def directory_mover(data,dir_name):

    co = 0

    for image in data.image_path:

        # create top directory

        if not os.path.exists(os.path.join('/kaggle/working/',dir_name)):

            shutil.os.mkdir(os.path.join('/kaggle/working/',dir_name))

        

        data_type = data[data['image_path'] == image]['name']

        data_type = str(list(data_type)[0])

        if not os.path.exists(os.path.join('/kaggle/working/',dir_name,data_type)):

            shutil.os.mkdir(os.path.join('/kaggle/working/',dir_name,data_type))

        path_from = os.path.join('../input/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/',image)

        path_to = os.path.join('/kaggle/working/',dir_name,data_type)

        # print(path_to)

        shutil.copy(path_from, path_to)

        # print('Moved {} to {}'.format(image,path_to))

        co += 1

        

    print('Moved {} images to {} folder.'.format(co,dir_name))
# move images:

directory_mover(bush_train,"Bush_train/")

directory_mover(bush_val,"Bush_val/")

directory_mover(bush_test,"Bush_test/")

train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale = 1./255,

shear_range = 0.2,

zoom_range = 0.2,

horizontal_flip = True)

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

Bush_training_set = train_datagen.flow_from_directory('/kaggle/working/Bush_train/',

                                                 target_size = (250, 250),

                                                 batch_size = 32,

                                                 class_mode = 'binary')

Bush_val_set = test_datagen.flow_from_directory('/kaggle/working/Bush_val/',

                                            target_size = (250, 250),

                                            batch_size = 32,

                                            class_mode = 'binary')

Bush_test_set = test_datagen.flow_from_directory('/kaggle/working/Bush_test/',

                                            target_size = (250, 250),

                                            batch_size = 32,

                                            class_mode = 'binary')
Bush_history = Bush_classifier.fit_generator(Bush_training_set,

                                             # set steps per epoch equal to the number of training images

                                             steps_per_epoch = 678,

                                             # there is a six hour runtime limit on Kaggle Kernels, restricting the maximum epoch number

                                             epochs = 8,

                                             validation_data = Bush_val_set,

                                             validation_steps = 100)
plt.plot(Bush_history.history['accuracy'])

plt.plot(Bush_history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()



plt.plot(Bush_history.history['loss'])

plt.plot(Bush_history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
from keras.preprocessing import image



# create a function to predict class of images in a directory, given a trained classifier

def predictions(dir_name,classifier,binary):

    # dir_name should include name of predicted invididual - could edit this to iterate through folders in future

    predictions = []

    for im in os.listdir(dir_name):

        test_image = image.load_img(dir_name + str(im), target_size = (250,250))

        test_image = image.img_to_array(test_image)

        test_image = np.expand_dims(test_image, axis = 0)

        if(binary is True):

            result = float(str(classifier.predict(test_image))[2])

        else: 

            result = np.argmax(classifier.predict(test_image))

        predictions.append(result)

    return(predictions)

# classes are passed through the algorithm in a random order - we want to retrieve the order of the classes

Bush_test_names = []

# collect all file names

for i in range(len(Bush_test_set.filenames)):

    Bush_test_names.append(Bush_test_set.filenames[i])

# extract unique names, in order

for i in range(len(Bush_test_names)):

    Bush_test_names[i] = Bush_test_names[i].split("/")[0]

Bush_test_name_order = list(OrderedDict.fromkeys(Bush_test_names))
# Compute predictions

Bush_predictions_0 = predictions("/kaggle/working/Bush_test/"+ Bush_test_name_order[0] + "/",Bush_classifier,binary = True)

Bush_predictions_1 = predictions("/kaggle/working/Bush_test/"+ Bush_test_name_order[1] + "/",Bush_classifier, binary = True)



# Compare predictions to actual classes

Bush_predictions_frame = pd.DataFrame(list(zip(Bush_predictions_0 + Bush_predictions_1,

                                                [0] * len(Bush_predictions_0) + [1] * len(Bush_predictions_1))),

                                       columns = ['Predictions','Actual'])
# function for finding the precision and recall of predictions, for any number of predictive classes

def prec_acc(predictions_frame):

    precision = []

    accuracy = []

    recall = []

    for i in range(len(set(predictions_frame.Predictions))):

        tp = predictions_frame[np.logical_and(predictions_frame['Actual'] == i, predictions_frame['Predictions'] == i)].shape[0]

        tn = predictions_frame[np.logical_and(predictions_frame['Actual'] != i, predictions_frame['Predictions'] != i)].shape[0]

        fp = predictions_frame[np.logical_and(predictions_frame['Actual'] != i, predictions_frame['Predictions'] == i)].shape[0]

        fn = predictions_frame[np.logical_and(predictions_frame['Actual'] == i, predictions_frame['Predictions'] != i)].shape[0]

        total_preds = predictions_frame.shape[0]

        precision.append(tp/(tp + fp))

        accuracy.append((tp + tn)/total_preds)

        recall.append(tp/(tp + fn))

    return(accuracy,precision,recall)
Bush_accuracy = prec_acc(Bush_predictions_frame)

print('Precision:' + str(Bush_accuracy[1]))

print('Recall:' + str(Bush_accuracy[2]))

print(Bush_test_name_order)
if "Bush_train" in os.listdir("./"):

    shutil.rmtree("./Bush_train")

if "Bush_val" in os.listdir("./"):

    shutil.rmtree("./Bush_val")

if "Bush_test" in os.listdir("./"):

    shutil.rmtree("./Bush_test")

# obtain 6 most common celebrities



image_paths['name'].value_counts()[:6]
multi_data = pd.concat([image_paths[image_paths.name=="George_W_Bush"].sample(75),

                        image_paths[image_paths.name=="Colin_Powell"].sample(75),

                        image_paths[image_paths.name=="Tony_Blair"].sample(75),

                        image_paths[image_paths.name=="Donald_Rumsfeld"].sample(75),

                        image_paths[image_paths.name=="Gerhard_Schroeder"].sample(75),

                        image_paths[image_paths.name=="Ariel_Sharon"].sample(75)])
# withhold final test data

multi_train, multi_test = train_test_split(multi_data, test_size=0.2)

# split into validation data

multi_train, multi_val = train_test_split(multi_train,test_size=0.2)



# Same model setup as before

multi_classifier = Sequential()

multi_classifier.add(Conv2D(32, (3, 3), input_shape = (250, 250, 3), activation = 'relu'))

multi_classifier.add(MaxPooling2D(pool_size = (2, 2)))

multi_classifier.add(Flatten())

multi_classifier.add(Dense(units = 128, activation = 'relu'))



# as we are training on several classes, we need several classification units (one for each class). We also use a 

# softmax activation function

multi_classifier.add(Dense(units = 6, activation = 'softmax'))

# we change the loss function to categorical_crossentropy

multi_classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])



# move images to separate directories

directory_mover(multi_train,"train_multi/")

directory_mover(multi_val,"val_multi/")

directory_mover(multi_test,"test_multi/")



train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale = 1./255,

shear_range = 0.2,

zoom_range = 0.2,

horizontal_flip = True)

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

multi_training_set = train_datagen.flow_from_directory('/kaggle/working/train_multi/',

                                                 target_size = (250, 250),

                                                 batch_size = 32,

                                                 class_mode = 'categorical')

multi_val_set = test_datagen.flow_from_directory('/kaggle/working/val_multi/',

                                            target_size = (250, 250),

                                            batch_size = 32,

                                            class_mode = 'categorical')

multi_test_set = test_datagen.flow_from_directory('/kaggle/working/test_multi/',

                                            target_size = (250, 250),

                                            batch_size = 32,

                                            class_mode = 'categorical')





multi_history = multi_classifier.fit_generator(multi_training_set,

                                               # set steps per epoch equal to the number of training images

                                               steps_per_epoch = 288,

                                               # as the data is smaller, we can run for more epochs

                                               epochs = 14,

                                               validation_data = multi_val_set,

                                               validation_steps = 72)
plt.plot(multi_history.history['accuracy'])

plt.plot(multi_history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()



plt.plot(multi_history.history['loss'])

plt.plot(multi_history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
# classes are passed through the algorithm in a random order - we want to retrieve the order of the classes

multi_test_names = []

# collect all file names

for i in range(len(multi_test_set.filenames)):

    multi_test_names.append(multi_test_set.filenames[i])

# extract unique names, in order

for i in range(len(multi_test_names)):

    multi_test_names[i] = multi_test_names[i].split("/")[0]

multi_test_name_order = list(OrderedDict.fromkeys(multi_test_names))
multi_predictions_0 = predictions("/kaggle/working/test_multi/" + multi_test_name_order[0] + "/",multi_classifier,binary=False)

multi_predictions_1 = predictions("/kaggle/working/test_multi/" + multi_test_name_order[1] + "/",multi_classifier,binary=False)

multi_predictions_2 = predictions("/kaggle/working/test_multi/" + multi_test_name_order[2] + "/",multi_classifier,binary=False)

multi_predictions_3 = predictions("/kaggle/working/test_multi/" + multi_test_name_order[3] + "/",multi_classifier,binary=False)

multi_predictions_4 = predictions("/kaggle/working/test_multi/" + multi_test_name_order[4] + "/",multi_classifier,binary=False)

multi_predictions_5 = predictions("/kaggle/working/test_multi/" + multi_test_name_order[5] + "/",multi_classifier,binary=False)

multi_predictions_frame = pd.DataFrame(list(zip(multi_predictions_0 + multi_predictions_1 + multi_predictions_2 + multi_predictions_3 + multi_predictions_4 + multi_predictions_5,

                                                [0] * len(multi_predictions_0) + [1] * len(multi_predictions_1) + [2] * len(multi_predictions_2) + [3] * len(multi_predictions_3) + [4] * len(multi_predictions_4) + [5] * len(multi_predictions_5))),

                                       columns = ['Predictions','Actual'])
multi_accuracy = prec_acc(multi_predictions_frame)

print('Precision:' + str(multi_accuracy[1]))

print('Recall:' + str(multi_accuracy[2]))

print(multi_test_name_order)
if "train_multi" in os.listdir("./"):

    shutil.rmtree("./train_multi")

if "val_multi" in os.listdir("./"):

    shutil.rmtree("./val_multi")

if "test_multi" in os.listdir("./"):

    shutil.rmtree("./test_multi")

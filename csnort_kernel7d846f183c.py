# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



### Install imutils

!pip install imutils #need to have internet on- see Settings



#### Import packages

import os

import numpy as np

import shutil

import cv2

import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np

import pickle

from imutils import paths

from keras.preprocessing.image import ImageDataGenerator

from keras.applications import VGG16

from keras.layers.core import Dropout

from keras.layers.core import Flatten

from keras.layers.core import Dense

from keras.layers import Input

from keras.models import Model

from keras.optimizers import SGD

from sklearn.metrics import classification_report
#CONFIG 



# initialize the base path to the *new* directory that will contain

# our images after computing the training and testing split

BASE_PATH = "/kaggle/working/finetuningkeras/dataset"



# define the names of the training, testing, and validation

# directories

TRAIN = "training"

TEST = "evaluation"

VAL = "validation"



REAL = 'REAL'

FAKE = 'FAKE'



# initialize the list of class label names

CLASSES = ["FAKE", "REAL"]





# set the batch size when fine-tuning

BATCH_SIZE = 32



trainEpochs = 10

epochsFineTune = 10

maxVids = 5



# set the path to the serialized model after training

MODEL_PATH = os.path.sep.join(["/kaggle/working/finetuningkeras","output", "Deepfake.model"])



# define the path to the output training history plots

UNFROZEN_PLOT_PATH = os.path.sep.join(["/kaggle/working/finetuningkeras","output", "unfrozen.png"])

WARMUP_PLOT_PATH = os.path.sep.join(["/kaggle/working/finetuningkeras","output", "warmup.png"])



file = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/metadata.json'

img_path = '/kaggle/input/deepfake-detection-challenge/train_sample_videos'

data_path = '/kaggle/working/finetuningkeras/real_fake'

dir_fake_frames = '/kaggle/working/FAKE_frames'

dir_real_frames = '/kaggle/working/REAL_frames'

dir_output = '/kaggle/working/finetuningkeras/output'



dir_data_path_real = os.path.join(data_path, REAL)

dir_data_path_fake = os.path.join(data_path, FAKE)



dir_train_real = os.path.join(BASE_PATH, TRAIN, REAL)

dir_train_fake = os.path.join(BASE_PATH, TRAIN, FAKE)

dir_valid_real = os.path.join(BASE_PATH, VAL, REAL)

dir_valid_fake = os.path.join(BASE_PATH, VAL, FAKE)

dir_test_real = os.path.join(BASE_PATH, TEST, REAL)

dir_test_fake = os.path.join(BASE_PATH, TEST, FAKE)



"""

Break mp4 files into individual images/frames (jpg)

input_dir: input/source directory containing one or more mp4 files

output_dir: target directory for saving individual frames (format: filename_frame#.jpg)

maxN: maximum number of mp4 files to explode

"""

input_dir = '/kaggle/working/finetuningkeras/real_fake/FAKE'

output_dir = '/kaggle/working/FAKE_frames/'

def explode_frames(input_dir, output_dir, maxN):



    mp4_filenames = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]

    n = 0

    

    for mp4fn in mp4_filenames:

        

        if(n < maxN):

            n += 1 

            mp4fp = os.path.join(input_dir, mp4fn)

            cam = cv2.VideoCapture(mp4fp) 

            if(cam.isOpened()):

                print('Processing file #'+ str(n) + ' (' + mp4fn + ')...')

            else: 

                print('Problem opening file #'+ str(n) + ' (' + mp4fn + ')...')

                continue 

            

            nframe = 0

            while(True): #continue until ret = False then break

                nframe += 1

                ret,frame = cam.read()

                

                if ret: 

                    # if video is still left continue creating images 

                    out_filename = os.path.splitext(mp4fn)[0]+  '_frame' + str(nframe) + '.jpg'

                    out_filepath =  os.path.join(output_dir, out_filename)

                    

                    # writing the extracted images 

                    cv2.imwrite(out_filepath, frame) 

                else: 

                    break



            # Release all space and windows once done

            print(' - created ' + str(nframe-1) + ' images') # -1 bc count incremented before exit

            cam.release() 

            cv2.destroyAllWindows()

            

        else: 

            break

            

"""

Distribute files/images from a source directory into training, validation, and testing directories. 

src_dir = source/input directory

train_dir, val_dir, test_dir = target training/validation/testing directory

valperc = fraction of dataset to use for validation (0-1)

testperc = fraction of dataset to use for testing (0-1)

"""

            

def trainvaltest_split(src_dir, train_dir, val_dir, test_dir, valperc = 0.15, testperc = 0.15):

    

    filenames = os.listdir(src_dir) #get all filenames in random order

    np.random.shuffle(filenames)

    

    n = len(filenames)

    split1 = int(n*(1 - (valperc + testperc)))

    split2 = int(n*(1 - (testperc)))

    

    fn_train, fn_val, fn_test = np.split(np.array(filenames), [split1, split2])

    

    fn_lists = [fn_train, fn_val, fn_test]

    targetdirs = [train_dir, val_dir, test_dir]

    

    print('Total images: ', n)

    print('Training: ', len(fn_train))

    print('Validation: ', len(fn_val))

    print('Testing: ', len(fn_test))

    

    all_fp = [os.path.join(src_dir, fn) for fn in filenames]

    

    #move files

    for i, fn_list in enumerate(fn_lists):

        for fn in fn_list: 

            target_dir = targetdirs[i]

            fp_from = os.path.join(src_dir, fn)

            fp_to = os.path.join(target_dir, fn)

            

            shutil.move(fp_from, fp_to)



            

"""

Construct a plot that plots and saves the training history

"""           

def plot_training(H, N, plotPath):

	plt.style.use("ggplot")

	plt.figure()

	plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")

	plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")

	plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")

	plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")

	plt.title("Training Loss and Accuracy")

	plt.xlabel("Epoch #")

	plt.ylabel("Loss/Accuracy")

	plt.legend(loc="lower left")

	plt.savefig(plotPath)
######### CREATING FOLDERS AND DATA STRUCTURE ########

# # Creating Train / Val / Test folders (One time use)



os.makedirs(dir_train_real, exist_ok = True)

os.makedirs(dir_train_fake, exist_ok = True)

os.makedirs(dir_valid_real, exist_ok = True)

os.makedirs(dir_valid_fake, exist_ok = True)

os.makedirs(dir_test_real, exist_ok = True)

os.makedirs(dir_test_fake, exist_ok = True)



os.makedirs(dir_data_path_real, exist_ok = True)

os.makedirs(dir_data_path_fake, exist_ok = True)

os.makedirs(dir_fake_frames, exist_ok = True) 

os.makedirs(dir_real_frames, exist_ok = True) 

os.makedirs(dir_output, exist_ok = True)
# Read in labels from json file

df = pd.read_json(file)

df = df.T



# %% [code]

label = df[['label']] #df with .mp4 filename index and label field
# copy training sample videos into /kaggle/working/finetuningkeras/real_fake/[FAKE or REAL]

# target dir



for fn, row in label.iterrows():

    src = os.path.join(img_path, fn)

    dest = os.path.join(data_path, row['label'], fn)

    shutil.copy(src, dest)
# Extract individual frames from fake vids

explode_frames(dir_data_path_fake, dir_fake_frames, maxN= maxVids)
# Extract individual frames from real vids

explode_frames(dir_data_path_real, dir_real_frames, maxN= maxVids)
## Split fake frames into training, validation, test sets

trainvaltest_split(src_dir = dir_fake_frames,

                   train_dir = dir_train_fake, 

                   val_dir = dir_valid_fake, 

                   test_dir = dir_test_fake)



## Split real frames into training, validation, test sets

trainvaltest_split(src_dir = dir_real_frames,

                   train_dir = dir_train_real, 

                   val_dir = dir_valid_real, 

                   test_dir = dir_test_real)
### IMAGE CLASSIFCATION ###

# initialize the training data augmentation object



trainAug = ImageDataGenerator(

	rotation_range=30,

	zoom_range=0.15,

	width_shift_range=0.2,

	height_shift_range=0.2,

	shear_range=0.15,

	horizontal_flip=True,

	fill_mode="nearest")
# initialize the validation/testing data augmentation object (which

# we'll be adding mean subtraction to)

valAug = ImageDataGenerator()
# define the ImageNet mean subtraction (in RGB order) and set the

# the mean subtraction value for each of the data augmentation

# objects

mean = np.array([123.68, 116.779, 103.939], dtype="float32")

trainAug.mean = mean

valAug.mean = mean
# ImageDataGenerator.flow_from_direcotry treats subdir names as classes



# initialize the training generator

trainPath = os.path.join(BASE_PATH, TRAIN)

trainGen = trainAug.flow_from_directory(

	trainPath,

	class_mode="categorical",

	target_size=(224, 224),

	color_mode="rgb",

	shuffle=True,

	batch_size=BATCH_SIZE)



# initialize the validation generator

valPath = os.path.join(BASE_PATH, VAL)

valGen = valAug.flow_from_directory(

	valPath,

	class_mode="categorical",

	target_size=(224, 224),

	color_mode="rgb",

	shuffle=False,

	batch_size=BATCH_SIZE)



# initialize the testing generator

testPath = os.path.join(BASE_PATH, TEST)

testGen = valAug.flow_from_directory(

	testPath,

	class_mode="categorical",

	target_size=(224, 224),

	color_mode="rgb",

	shuffle=False,

	batch_size=BATCH_SIZE)
# load the VGG16 network, ensuring the head FC layer sets are left off

baseModel = VGG16(weights="imagenet", include_top=False,

	input_tensor=Input(shape=(224, 224, 3)))
# construct the head of the model that will be placed on top of the

# the base model

headModel = baseModel.output

headModel = Flatten(name="flatten")(headModel)

headModel = Dense(512, activation="relu")(headModel)

headModel = Dropout(0.5)(headModel)

headModel = Dense(len(CLASSES), activation="softmax")(headModel)
# place the head FC model on top of the base model (this will become

# the actual model we will train)

model = Model(inputs=baseModel.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will

# *not* be updated during the first training process

for layer in baseModel.layers:

	layer.trainable = False
# compile our model (this needs to be done after our setting our

# layers to being non-trainable

print("[INFO] compiling model...")

opt = SGD(lr=1e-4, momentum=0.9)

model.compile(loss="categorical_crossentropy", optimizer=opt,

	metrics=["accuracy"])
# train the head of the network for a few epochs (all other layers

# are frozen) -- this will allow the new FC layers to start to become

# initialized with actual "learned" values versus pure random

# and testing directories

totalTrain = len(list(paths.list_images(trainPath)))

totalVal = len(list(paths.list_images(valPath)))

totalTest = len(list(paths.list_images(testPath)))



print("[INFO] training head...")

H = model.fit_generator(

	trainGen,

	steps_per_epoch=totalTrain // BATCH_SIZE,

	validation_data=valGen,

	validation_steps=totalVal // BATCH_SIZE,

	epochs= trainEpochs)
# reset the testing generator and evaluate the network after

# fine-tuning just the network head

print("[INFO] evaluating after fine-tuning network head...")

testGen.reset()

predIdxs = model.predict_generator(testGen,

	steps=(totalTest // BATCH_SIZE) + 1)

predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(testGen.classes, predIdxs,

	target_names=testGen.class_indices.keys()))





plot_training(H, trainEpochs, WARMUP_PLOT_PATH)

plt.show()  #plot not showning in kaggle notebook.. does it support ploting?
# reset our data generators

trainGen.reset()

valGen.reset()
# now that the head FC layers have been trained/initialized, lets

# unfreeze the final set of CONV layers and make them trainable

for layer in baseModel.layers[15:]:

	layer.trainable = True



# loop over the layers in the model and show which ones are trainable

# or not

for layer in baseModel.layers:

	print("{}: {}".format(layer, layer.trainable))



# for the changes to the model to take affect we need to recompile

# the model, this time using SGD with a *very* small learning rate

print("[INFO] re-compiling model...")

opt = SGD(lr=1e-4, momentum=0.9)

model.compile(loss="categorical_crossentropy", optimizer=opt,

	metrics=["accuracy"])
# train the model again, this time fine-tuning *both* the final set

# of CONV layers along with our set of FC layers

H = model.fit_generator(

	trainGen,

	steps_per_epoch=totalTrain // BATCH_SIZE,

	validation_data=valGen,

	validation_steps=totalVal // BATCH_SIZE,

	epochs= epochsFineTune)
# reset the testing generator and then use our trained model to

# make predictions on the data

print("[INFO] evaluating after fine-tuning network...")

testGen.reset()

predIdxs = model.predict_generator(testGen,

	steps=(totalTest // BATCH_SIZE) + 1)

predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(testGen.classes, predIdxs,

	target_names=testGen.class_indices.keys()))

plot_training(H, epochsFineTune, UNFROZEN_PLOT_PATH)



# serialize the model to disk

print("[INFO] serializing network...")

model.save(MODEL_PATH)
### fast.ai section ####
from fastai.vision import *

from fastai.metrics import error_rate


tfms = get_transforms(do_flip=False)





# fastai subdirs need to be named train, valid, test (test is optional)

os.rename('/kaggle/working/finetuningkeras/dataset/training', 

          '/kaggle/working/finetuningkeras/dataset/train')

os.rename('/kaggle/working/finetuningkeras/dataset/validation', 

          '/kaggle/working/finetuningkeras/dataset/valid')

os.rename('/kaggle/working/finetuningkeras/dataset/evaluation', 

          '/kaggle/working/finetuningkeras/dataset/test')





data = ImageDataBunch.from_folder(BASE_PATH, ds_tfms=tfms, size=224)

data.show_batch(rows=3, figsize=(5,5))
learn = cnn_learner(data, models.resnet18, metrics=accuracy)
learn.fit(2)
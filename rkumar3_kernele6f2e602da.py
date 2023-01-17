# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


 

from __future__ import print_function



# Networks

from keras.preprocessing import image

from keras.applications.resnet50 import ResNet50

from keras.applications.vgg16 import VGG16

from keras.applications.vgg19 import VGG19

from keras.applications.inception_v3 import InceptionV3

from keras.applications.xception import Xception

from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.applications.mobilenet import MobileNet

from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201

from keras.applications.nasnet import NASNetLarge, NASNetMobile

from keras.preprocessing.image import ImageDataGenerator



# Layers

from keras.layers import Dense, Activation, Flatten, Dropout

from keras import backend as K



# Other

from keras import optimizers

from keras import losses

from keras.optimizers import SGD, Adam

from keras.models import Sequential, Model

from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from keras.models import load_model



# Utils

import matplotlib.pyplot as plt

import numpy as np

import argparse

import random, glob

import os, sys, csv

import cv2

import time, datetime



from keras.layers import Dense, Activation, Flatten, Dropout

from keras import backend as K



# Other

from keras import optimizers

from keras import losses

from keras.optimizers import SGD, Adam

from keras.models import Sequential, Model

from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from keras.models import load_model



# Utils

import matplotlib.pyplot as plt

import numpy as np

import argparse

import random, glob

import os, sys, csv

import cv2

import time, datetime





# Files

#import utils

####################################

def save_class_list(class_list, model_name, dataset_name):

    class_list.sort()

    with open("./checkpoints/" + model_name + "_" + dataset_name + "_class_list.txt",'wb') as target:

        for c in class_list:

            target.write(c.encode())

            target.write("\n".encode())



def load_class_list(class_list_file):

    class_list = []

    with open(class_list_file, 'r') as csvfile:

        file_reader = csv.reader(csvfile)

        for row in file_reader:

            class_list.append(row)

    class_list.sort()

    return class_list



# Get a list of subfolders in the directory

def get_subfolders(directory):

    subfolders = os.listdir(directory)

    subfolders.sort()

    return subfolders



# Get number of files by searching directory recursively

def get_num_files(directory):

    if not os.path.exists(directory):

        return 0

    cnt = 0

    for r, dirs, files in os.walk(directory):

        for dr in dirs:

            cnt += len(glob.glob(os.path.join(r, dr + "/*")))

    return cnt



# Add on new FC layers with dropout for fine tuning

def build_finetune_model(base_model, dropout, fc_layers, num_classes):

    for layer in base_model.layers:

        layer.trainable = False



    x = base_model.output

    x = Flatten()(x)

    for fc in fc_layers:

        x = Dense(fc, activation='relu')(x) # New FC layer, random init

        x = Dropout(dropout)(x)



    predictions = Dense(num_classes, activation='softmax')(x) # New softmax layer

    

    finetune_model = Model(inputs=base_model.input, outputs=predictions)



    return finetune_model



# Plot the training and validation loss + accuracy

def plot_training(history):

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    epochs = range(len(acc))



    plt.plot(epochs, acc, 'r.')

    plt.plot(epochs, val_acc, 'r')

    plt.title('Training and validation accuracy')



    # plt.figure()

    # plt.plot(epochs, loss, 'r.')

    # plt.plot(epochs, val_loss, 'r-')

    # plt.title('Training and validation loss')

    plt.show()



    plt.savefig('acc_vs_epochs.png')





####################################



# For boolean input from the command line

def str2bool(v):

    if v.lower() in ('yes', 'true', 't', 'y', '1'):

        return True

    elif v.lower() in ('no', 'false', 'f', 'n', '0'):

        return False

    else:

        raise argparse.ArgumentTypeError('Boolean value expected.')





# Command line args

# parser = argparse.ArgumentParser()

# parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train for')

# parser.add_argument('--mode', type=str, default="train", help='Select "train", or "predict" mode. \

#     Note that for prediction mode you have to specify an image to run the model on.')

# parser.add_argument('--image', type=str, default=None, help='The image you want to predict on. Only valid in "predict" mode.')

# parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')

# parser.add_argument('--dataset', type=str, default="Pets", help='Dataset you are using.')

# parser.add_argument('--resize_height', type=int, default=224, help='Height of cropped input image to network')

# parser.add_argument('--resize_width', type=int, default=224, help='Width of cropped input image to network')

# parser.add_argument('--batch_size', type=int, default=32, help='Number of images in each batch')

# parser.add_argument('--dropout', type=float, default=1e-3, help='Dropout ratio')

# parser.add_argument('--h_flip', type=str2bool, default=False, help='Whether to randomly flip the image horizontally for data augmentation')

# parser.add_argument('--v_flip', type=str2bool, default=False, help='Whether to randomly flip the image vertically for data augmentation')

# parser.add_argument('--rotation', type=float, default=0.0, help='Whether to randomly rotate the image for data augmentation')

# parser.add_argument('--zoom', type=float, default=0.0, help='Whether to randomly zoom in for data augmentation')

# parser.add_argument('--shear', type=float, default=0.0, help='Whether to randomly shear in for data augmentation')

# parser.add_argument('--model', type=str, default="MobileNet", help='Your pre-trained classification model of choice')

# args = parser.parse_args()





# Global settings

num_epochs = 10

mode = 'train'

dataset = '/kaggle/input/fruits-vegetables-photos/'

dropout = 0.001

continue_training = False

image = None



BATCH_SIZE = 32

WIDTH = 224

HEIGHT = 224

FC_LAYERS = [1024, 1024]

TRAIN_DIR = "/kaggle/input/fruits-vegetables-photos/TRAIN/"

VAL_DIR = "/kaggle/input/fruits-vegetables-photos/TEST/"

model = "ResNet50"



preprocessing_function = None

base_model = None







# Prepare the model

if model == "VGG16":

    from keras.applications.vgg16 import preprocess_input

    preprocessing_function = preprocess_input

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))

elif model == "VGG19":

    from keras.applications.vgg19 import preprocess_input

    preprocessing_function = preprocess_input

    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))

elif model == "ResNet50":

    from keras.applications.resnet50 import preprocess_input

    preprocessing_function = preprocess_input

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))

elif model == "InceptionV3":

    from keras.applications.inception_v3 import preprocess_input

    preprocessing_function = preprocess_input

    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))

elif model == "Xception":

    from keras.applications.xception import preprocess_input

    preprocessing_function = preprocess_input

    base_model = Xception(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))

elif model == "InceptionResNetV2":

    from keras.applications.inceptionresnetv2 import preprocess_input

    preprocessing_function = preprocess_input

    base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))

elif model == "MobileNet":

    from keras.applications.mobilenet import preprocess_input

    preprocessing_function = preprocess_input

    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))

elif model == "DenseNet121":

    from keras.applications.densenet import preprocess_input

    preprocessing_function = preprocess_input

    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))

elif model == "DenseNet169":

    from keras.applications.densenet import preprocess_input

    preprocessing_function = preprocess_input

    base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))

elif model == "DenseNet201":

    from keras.applications.densenet import preprocess_input

    preprocessing_function = preprocess_input

    base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))

elif model == "NASNetLarge":

    from keras.applications.nasnet import preprocess_input

    preprocessing_function = preprocess_input

    base_model = NASNetLarge(weights='imagenet', include_top=True, input_shape=(HEIGHT, WIDTH, 3))

elif model == "NASNetMobile":

    from keras.applications.nasnet import preprocess_input

    preprocessing_function = preprocess_input

    base_model = NASNetMobile(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))

else:

    ValueError("The model you requested is not supported in Keras")



    

if mode == "train":

    print("\n***** Begin training *****")

    print("Dataset -->", dataset)

    print("Model -->", model)

    print("Resize Height -->", HEIGHT)

    print("Resize Width -->", WIDTH)

    print("Num Epochs -->", num_epochs)

    print("Batch Size -->", BATCH_SIZE)



    print("Data Augmentation:")

    print("\tRotation -->", 0.0)

    print("\tZooming -->", 0.0)

    print("\tShear -->", 0.0)

    print("")



    # Create directories if needed

    if not os.path.isdir("checkpoints"):

        os.makedirs("checkpoints")



    # Prepare data generators

    train_datagen =  ImageDataGenerator(

      preprocessing_function=preprocessing_function,

      rotation_range=0.0,

      shear_range=0.0,

      zoom_range=0.0,

      horizontal_flip=True,

      vertical_flip=True

    )



    val_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)



    train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE)



    validation_generator = val_datagen.flow_from_directory(VAL_DIR, target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE)





    # Save the list of classes for prediction mode later

    class_list = get_subfolders(TRAIN_DIR)

    save_class_list(class_list, model_name=model, dataset_name=dataset.split('/')[-1])



    finetune_model = build_finetune_model(base_model, dropout=dropout, fc_layers=FC_LAYERS, num_classes=len(class_list))



    if continue_training:

        finetune_model.load_weights("./checkpoints/" + model + "_model_weights.h5")



    adam = Adam(lr=0.00001)

    finetune_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])



    num_train_images = get_num_files(TRAIN_DIR)

    num_val_images = get_num_files(VAL_DIR)



    def lr_decay(epoch):

        if epoch%20 == 0 and epoch!=0:

            lr = K.get_value(model.optimizer.lr)

            K.set_value(model.optimizer.lr, lr/2)

            print("LR changed to {}".format(lr/2))

        return K.get_value(model.optimizer.lr)



    learning_rate_schedule = LearningRateScheduler(lr_decay)



    filepath="./checkpoints/" + model + "_model_weights.h5"

    checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')

    callbacks_list = [checkpoint]





    history = finetune_model.fit_generator(train_generator, epochs=num_epochs, workers=8, steps_per_epoch=num_train_images // BATCH_SIZE, 

        validation_data=validation_generator, validation_steps=num_val_images // BATCH_SIZE, class_weight='auto', shuffle=True, callbacks=callbacks_list)





    plot_training(history)



elif mode == "predict":



    if image is None:

        ValueError("You must pass an image path when using prediction mode.")



    # Create directories if needed

    if not os.path.isdir("%s"%("Predictions")):

        os.makedirs("%s"%("Predictions"))



    # Read in your image

    image = cv2.imread(image,-1)

    save_image = image

    image = np.float32(cv2.resize(image, (HEIGHT, WIDTH)))

    image = preprocessing_function(image.reshape(1, HEIGHT, WIDTH, 3))



    class_list_file = "./checkpoints/" + model + "_" + dataset + "_class_list.txt"



    class_list = load_class_list(class_list_file)

    

    finetune_model = build_finetune_model(base_model, len(class_list))

    finetune_model.load_weights("./checkpoints/" + model + "_model_weights.h5")



    # Run the classifier and print results

    st = time.time()



    out = finetune_model.predict(image)



    confidence = out[0]

    class_prediction = list(out[0]).index(max(out[0]))

    class_name = class_list[class_prediction]



    run_time = time.time()-st



    print("Predicted class = ", class_name)

    print("Confidence = ", confidence)

    print("Run time = ", run_time)

    cv2.imwrite("Predictions/" + class_name[0] + ".png", save_image)



    from IPython.display import FileLink

    FileLink(r'ResNet50_model_weights.h5')
!ls
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download h5 file", filename = "ResNet50_model_weights.h5"):  

    csv = df

    b64 = base64.b64encode(csv)

    payload = b64.decode()

    html = '<a download="{filename}" href="data:h5,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)
with open('ResNet50_model_weights.h5', 'rb') as f:

    a= f.read()
create_download_link(a)
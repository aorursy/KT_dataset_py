# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings("ignore")



from keras.applications import MobileNet

from keras.layers import Dense,Dropout,GlobalAveragePooling2D

from keras.models import Model



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
mobilnet=MobileNet(weights="imagenet",include_top=False,input_shape=(224,224,3))
#Freeze layers

for layer in mobilnet.layers:

    layer.trainable=False

    

for (i,layer) in enumerate(mobilnet.layers):

    print(str(i)+" "+layer.__class__.__name__,layer.trainable)
def addTopModelMobileNet(bottom_model,num_classes):

    top_model=bottom_model.output

    top_model=GlobalAveragePooling2D()(top_model)

    top_model=Dense(1024,activation="relu")(top_model)

    top_model=Dense(1024,activation="relu")(top_model)

    top_model=Dense(512,activation="relu")(top_model)

    top_model=Dense(num_classes,activation="softmax")(top_model)

    

    return top_model
num_classes = 10



Fc_head=addTopModelMobileNet(mobilnet,num_classes)



model=Model(inputs=mobilnet.input,outputs=Fc_head)



print(model.summary())
from keras.preprocessing.image import ImageDataGenerator



train_data_dir = '/kaggle/input/10-monkey-species/training/training'

validation_data_dir = '/kaggle/input/10-monkey-species/validation/validation'



trainDataGen=ImageDataGenerator(rescale=1./255,

                               rotation_range=45,

                               width_shift_range=0.3,

                               height_shift_range=0.3,

                               horizontal_flip=True,

                               fill_mode="nearest")



validationDataGen=ImageDataGenerator(rescale=1./255)



batch_size=32



trainGenerator=trainDataGen.flow_from_directory(train_data_dir,

                                               target_size=(224,224),

                                               batch_size=batch_size,

                                               class_mode="categorical")



validationGenerator=validationDataGen.flow_from_directory(validation_data_dir,

                                                         target_size=(224,224),

                                                         batch_size=batch_size,

                                                         class_mode="categorical")
from keras.optimizers import RMSprop

from keras.callbacks import EarlyStopping,ModelCheckpoint



checkpoint = ModelCheckpoint("monkey_breed_mobileNet.h5",

                             monitor="val_loss",

                             mode="min",

                             save_best_only = True,

                             verbose=1)



earlystop = EarlyStopping(monitor = 'val_loss', 

                          min_delta = 0, 

                          patience = 3,

                          verbose = 1,

                          restore_best_weights = True)



# we put our call backs into a callback list

callbacks = [earlystop, checkpoint]



model.compile(loss="categorical_crossentropy",optimizer=RMSprop(lr=0.001),metrics=["accuracy"])



epochs=10

batch_size=16



history = model.fit_generator(

    trainGenerator,

    steps_per_epoch = trainGenerator.samples//batch_size,

    epochs = epochs,

    callbacks = callbacks,

    validation_data = validationGenerator,

    validation_steps = validationGenerator.samples//batch_size)
import os

import cv2

import numpy as np

from os import listdir

from os.path import isfile, join

import matplotlib.pyplot as plt



monkey_breeds_dict = {"[0]": "mantled_howler ", 

                      "[1]": "patas_monkey",

                      "[2]": "bald_uakari",

                      "[3]": "japanese_macaque",

                      "[4]": "pygmy_marmoset ",

                      "[5]": "white_headed_capuchin",

                      "[6]": "silvery_marmoset",

                      "[7]": "common_squirrel_monkey",

                      "[8]": "black_headed_night_monkey",

                      "[9]": "nilgiri_langur"}



monkey_breeds_dict_n = {"n0": "mantled_howler ", 

                      "n1": "patas_monkey",

                      "n2": "bald_uakari",

                      "n3": "japanese_macaque",

                      "n4": "pygmy_marmoset ",

                      "n5": "white_headed_capuchin",

                      "n6": "silvery_marmoset",

                      "n7": "common_squirrel_monkey",

                      "n8": "black_headed_night_monkey",

                      "n9": "nilgiri_langur"}



def draw_test(name, pred, im):

    monkey = monkey_breeds_dict[str(pred)]

    BLACK = [0,0,0]

    #expanded_image = cv2.copyMakeBorder(im, 80, 0, 0, 100 ,cv2.BORDER_CONSTANT,value=BLACK)

    plt.imshow(im)

    plt.title(monkey)

    plt.show()

    #cv2.putText(expanded_image, monkey, (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)

    #cv2.imshow(name, expanded_image)



def getRandomImage(path):

    """function loads a random images from a random folder in our test path """

    folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))

    random_directory = np.random.randint(0,len(folders))

    path_class = folders[random_directory]

    print("Class - " + monkey_breeds_dict_n[str(path_class)])

    file_path = path + path_class

    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]

    random_file_index = np.random.randint(0,len(file_names))

    image_name = file_names[random_file_index]

    return cv2.imread(file_path+"/"+image_name)    



for i in range(0,10):

    input_im = getRandomImage("/kaggle/input/10-monkey-species/validation/validation/")

    input_original = input_im.copy()

    input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)

    

    input_im = cv2.resize(input_im, (224, 224), interpolation = cv2.INTER_LINEAR)

    input_im = input_im / 255.

    input_im = input_im.reshape(1,224,224,3) 

    

    # Get Prediction

    res = np.argmax(model.predict(input_im, 1, verbose = 0), axis=1)

    

    # Show image with predicted class

    draw_test("Prediction", res, input_original) 

    #cv2.waitKey(0)



#cv2.destroyAllWindows()
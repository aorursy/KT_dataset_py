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
os.listdir("../")
if not os.path.exists("../train"):

    os.mkdir("../train")

if not os.path.exists("../train/dog"):

    os.mkdir("../train/dog")

if not os.path.exists("../train/cat"):

    os.mkdir("../train/cat")
import shutil

from tqdm import tqdm
for i in tqdm(os.listdir("../input/train/train")):

#     print(i)

    if i.split(".")[0] == "dog":

        shutil.copy2(os.path.join("../input/train/train",i),os.path.join("../train/dog",i))

    elif i.split(".")[0] == "cat":

        shutil.copy2(os.path.join("../input/train/train",i),os.path.join("../train/cat",i))

        
if not os.path.exists("../test"):

    os.mkdir("../test")

if not os.path.exists("../test/dog"):

    os.mkdir("../test/dog")

if not os.path.exists("../test/cat"):

    os.mkdir("../test/cat")
for i in tqdm(os.listdir("../train/dog")[:3000]):

    shutil.move(os.path.join("../train/dog",i),os.path.join("../test/dog",i)) 

for i in tqdm(os.listdir("../train/cat")[:3000]):

    shutil.move(os.path.join("../train/cat",i),os.path.join("../test/cat",i)) 
import keras

from keras.models import Model

from keras.layers import Dense

from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing import image
os.listdir("../train")
trdata = ImageDataGenerator()

traindata = trdata.flow_from_directory(directory="../train",target_size=(224,224))
tsdata = ImageDataGenerator()

testdata = tsdata.flow_from_directory(directory="../test", target_size=(224,224))
from keras.applications.vgg16 import VGG16
vggmodel = VGG16(weights='imagenet', include_top=True)
vggmodel.summary()
for layers in (vggmodel.layers)[:19]:

    print(layers)

    layers.trainable = False
X= vggmodel.layers[-2].output
predictions = Dense(2, activation="softmax")(X)
model_final = Model(input = vggmodel.input, output = predictions)
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
model_final.summary()
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
os.listdir("../")
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

early = EarlyStopping(monitor='val_acc', min_delta=0, patience=40, verbose=1, mode='auto')

hist = model_final.fit_generator(generator= traindata, steps_per_epoch= 2, epochs= 100, validation_data= testdata, validation_steps=1, callbacks=[checkpoint,early])
model_final.save_weights("vgg16_1.h5")
import pandas as pd

df=pd.read_csv("../input/sampleSubmission.csv")
print(df["label"][0])

pd.options.mode.chained_assignment = None  # default='warn'
for e,i in enumerate(os.listdir("../input/test1/test1")):

    print(i)

    output=[]

    img = image.load_img(os.path.join("../input/test1/test1",i),target_size=(224,224))

    img = np.asarray(img)

    img = np.expand_dims(img, axis=0)

    output = model_final.predict(img)

    if output[0][0] > output[0][1]:

#         print("cat")

        df["id"][e]=i

        df["label"][e]="cat"

    else:

#         print('dog')

        df["id"][e]=i

        df["label"][e]="dog"
df
df.to_csv("submission.csv",index=False)
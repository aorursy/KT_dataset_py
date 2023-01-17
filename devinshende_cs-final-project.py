import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from time import time,sleep

from termcolor import cprint, colored

from random import randint, choice

import cv2

from tqdm import tqdm

import os

import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model



os.listdir("../input/dogs-vs-cats-redux-kernels-edition")
# class I use later on to time things that take a long time

class stopwatch():

    def __init__(self, msg=None):

        self.t0 = time()

        if msg == None:

            self.msg = ""

        else:

            self.msg = msg

    

    def stop(self):

        self.elapsed = round(time() - self.t0,1)

    

        # the math is from stack overflow

        hours = self.elapsed // 3600 % 24

        minutes = self.elapsed // 60 % 60

        seconds = self.elapsed % 60

    

        hours = round(hours)

        minutes = round(minutes)

        seconds = round(seconds,1)



        if self.msg:

            m = "Elapsed Time For "+self.msg+": "

        else:

            m = "Elapsed Time: "

        if hours and minutes and seconds:

            msg = f"{m}{hours} hours, {minutes} minutes, and {seconds} seconds"

        elif minutes and seconds:

            if minutes == 1.0:

                msg = f"{m}1 minute and {seconds} seconds"

            else: msg = f"{m}{minutes} minutes and {seconds} seconds"

        elif seconds:

            msg = f"{m}{seconds} seconds"

        elif seconds == 0.0:

            msg = f"{m}{round(time()-self.t0,6)} seconds" # give time rounded to 6 places (not 1) if it is 0 when rounded. Ex: 0.000143 seconds

        else: 

            msg = "something went wrong with stopwatch class\n"+str(self.elapsed)+" seconds"

        cprint(msg,"red")
IMSIZE = 50

path = "../input/dogs-vs-cats-redux-kernels-edition/train/"

def load_image(path):

    try:

        img = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), # read as grayscale

                            (IMSIZE, IMSIZE)) # resize it to IMG_SIZE

    except:

        print("error loading an image")

        return 

    return img

    

x = load_image(path+"dog."+str(randint(0,2000-1))+".jpg")

plt.imshow(x)

plt.title("a doggo")

plt.axis('off')

plt.show()
def create_train_data(stop=100):

    error = False

    print(f"Loading {stop} images")

    if stop == "all": stop = -1

    images = []

    for i in tqdm(os.listdir(path)[:stop]):

        error = False

        im = load_image(path+i)

        try:

            if im == None:

                error = True

                print("error loading an image")

        except:

                images.append(im)

        

    labels = []

    for i in os.listdir(path)[:stop]:

        if i[:3] == "cat": 

            if not error: 

                labels.append(np.array(

                    [1,0] # very cat, no doggo

                ))

        elif i[:3] == "dog": 

            if not error: 

                labels.append(np.array(

                    [0,1] # no cat, very doggo

                ))

        else: print("a problem occured in labelling")

    return images, labels



def show_distribution(labels):

    c = 0

    d = 0

    for i in labels:

        if list(i) == [1,0]:

            c += 1

        elif list(i) == [0,1]:

            d += 1

    print(f"cats: {c}, dogs: {d}")



images, labels = create_train_data(10000)

images = np.array(images) # because why not

labels = np.array(labels) # and also keras expects np arrays

print([len(i) for i in [images, labels]])

show_distribution(labels)

plt.imshow(images[2],cmap='gray')

plt.axis('off')

plt.show()
index = int(len(labels) * 0.1)



x_train = np.array(images[:-index]).reshape(-1,IMSIZE,IMSIZE,3) / 255.0 # normalize

y_train = np.array(labels[:-index])



x_val = np.array(images[-index:]).reshape(-1,IMSIZE,IMSIZE,3) / 255.0 # normalize

y_val = np.array(labels[-index:])



print([len(i) for i in [x_train, y_train, x_val, y_val]])

print("Shape of input data: ",x_train.shape)
from tensorflow.keras import Sequential

from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D



model = Sequential([

    Conv2D(32, (5,5), strides=(1,1) ,input_shape=x_train[0].shape, name='conv1'),

    MaxPooling2D(pool_size=(2,2), name='pool1' ),

    Conv2D(64, (5,5), name='conv2', activation='relu'),

    MaxPooling2D((2,2), name='pool2'),

    Conv2D(4, (2,2), name='conv3'),

    Flatten(name='flatten'),

    Dense(256, activation='relu', name='dense1'),

    Dropout(0.2),

    Dense(100, activation='relu', name='dense2'),

    Dense(2, activation='softmax', name='output')

])



model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

model.summary()
m_path = "../input/a-cnn-model-70/"

state = "train it"

if os.path.exists(m_path+'my_model.h5') and state=="normal":

    print(os.listdir(m_path))

    print("loading model from file...")

    print()

    model = load_model(m_path+"my_model.h5")

    model.summary()

else: 

    print("training model")

    t = stopwatch("Training")

    model.fit(x_train, y_train, epochs=5, validation_data=(x_val,y_val))

    print()

    t.stop()

    model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
from termcolor import colored

try:

    print(colored("Accuracy: ","cyan"),round(model.history.history['acc'][-1]*100,2),"%")

    print(colored("Validation Accuracy: ","cyan"),round(model.history.history['val_acc'][-1]*100,2),"%")



    plt.plot(model.history.history["val_acc"],label="val")

    plt.plot(model.history.history["acc"],label="train")

    plt.title("accuracy over time")

    plt.legend()

    plt.show()



    plt.plot(model.history.history["val_loss"],label="val")

    plt.plot(model.history.history["loss"],label="train")

    plt.title("loss over time")

    plt.legend()

    plt.show()

except AttributeError:

#     print("Loaded pretrained model, so can't show graph of progress :(")

    print("evaluating accuracy...")

    val_acc = round(model.evaluate(x_val, y_val)[0]*100,2)

#     print(val_acc)

    print(colored("Validation Accuracy: ","cyan"),f"{val_acc}%")
def makeplot():

    fig, a = plt.subplots(2,2)

    listofimages = []

    for r in range(2):

        for c in range(2):

            i = randint(0,len(x_val)-1)

            listofimages.append(i)

            if i in listofimages: i = randint(0,len(x_val)-1)

            a[r, c].imshow(x_val[i])

            a[r, c].axis('off')

            catness, dogness = predictions[i]

            if dogness > catness:

                percent = str(round(dogness*100,2))+"% dog"

            elif catness > dogness:

                percent = str(round(catness*100,2))+"% cat"

            a[r,c].set_title(percent)

            # show if it got it right or not

            if list( y_val[i] ) == [round(catness), round(dogness)]: 

                a[r,c].text(3, 8, 'correct', style='italic',

                bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 2})

            else: 

                a[r,c].text(3, 8, 'incorrect', style='italic',

                bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 2})

                a[r,c].set_xlabel('incorrect')



    plt.show()
predictions = model.predict(x_val)

first = list(predictions[0])

same = True

print("first few predictions\n",predictions[:4])

for i in predictions:

    if list(i) != first:

        same = False

if not same:

    for i in range(5):

        makeplot()



else:

    print("all predictions are the same :(")

    print("They are all ",predictions[0])
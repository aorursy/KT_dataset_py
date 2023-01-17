# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random

from scipy import ndarray

import skimage as sk

from skimage import transform

from skimage import util

from copy import deepcopy



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

import keras

from keras import backend as K

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

import cv2  # for image processing

import scipy.io

import os

#import keras_utils

#from keras_utils import reset_tf_session 

print(tf.__version__)

print(keras.__version__)
plt.rcParams.update({'axes.titlesize': 'small'})
IMG_SIZE = 128
import glob, pandas as pd
!ls ../input/
!ls ../input/aptos2019-blindness-detection
pd.read_csv("../input/aptos2019-blindness-detection/train.csv").head()
labels_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv").set_index('id_code')
len(labels_df.index)
labels_df['diagnosis'].value_counts()
images = [f for f in glob.glob("../input/aptos2019-blindness-detection/train_images/" + "*.png")]

labels = [ labels_df.loc[f.split('/')[-1].split('.')[0].strip()].diagnosis for f in  images]
train_images_ixs = set(random.choices(range(len(images)), k=6000))

train_images = []

train_labels = []

test_images = []

test_labels = []

for i in range(len(labels)):

    if i in train_images_ixs:

        train_images.append(images[i])

        train_labels.append(labels[i])

    else:

        test_images.append(images[i])

        test_labels.append(labels[i])
#print (train_images_ixs)

print (len(train_labels), len(train_images))

print (len(test_labels), len(test_images))
#test_labels_df = pd.read_csv("../input/test.csv").set_index('id_code')
for img,lb in zip(images[:5], labels[:5]):

    print (img, lb)
from PIL import Image

from matplotlib import pyplot as plt

import random
label_text = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
images_to_display = []

for lb in range(5):

    images_to_display += random.choices([ (images[ix], labels[ix]) for ix in range(len(images)) if labels[ix] == lb ] , k=10)



fig = plt.figure(figsize=(25, 16))

for ii, (img,label) in enumerate(images_to_display):

    ax = fig.add_subplot(5, 10, ii + 1, xticks=[], yticks=[])

    img = cv2.imread(img)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #img = Image.open(img_byte)

    plt.imshow(img)

    plt.text(0, img.shape[0], label_text[label], bbox=dict(facecolor='red', alpha=0.5))
def adjust_gamma(image, gamma=1.0):

    # build a lookup table mapping the pixel values [0, 255] to

    # their adjusted gamma values

    invGamma = 1.0 / gamma

    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table

    return cv2.LUT(image, table)
def add_contrast(img, contrast):

        buf = img.copy()

        f = float(131 * (contrast + 127)) / (127 * (131 - contrast))

        alpha_c = f

        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf
def preproces_image(img):

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = adjust_gamma(img, 1.5)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = add_contrast(img, 20)

    #laplacian = cv2.Laplacian(img,cv2.CV_64F)

    #sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)

    #sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

    return img

    
fig = plt.figure(figsize=(25, 16))

for ii, (img,label) in enumerate(images_to_display):

    ax = fig.add_subplot(5, 10, ii + 1, xticks=[], yticks=[])

    img = cv2.imread(img)

    img_new = preproces_image(img)

    #print (img.shape)

    #img = Image.open(img_byte)

    plt.imshow(img_new)

    plt.text(0, img_new.shape[0], label_text[label], bbox=dict(facecolor='red', alpha=0.5))
def random_rotation(image_array: ndarray):

    # pick a random degree of rotation between 25% on the left and 25% on the right

    random_degree = random.uniform(-180, 180)

    return sk.transform.rotate(image_array, random_degree)



def random_noise(image_array: ndarray):

    # add random noise to the image

    return sk.util.random_noise(image_array)



def horizontal_flip(image_array: ndarray):

    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !

    return image_array[:, ::-1]
wts = [0.1, 0.4, 0.2 , 0.90, 0.60]
img = cv2.imread(train_images[0])

img = preproces_image(img)

img = cv2.resize(img, (IMG_SIZE, IMG_SIZE) )

plt.imshow(img)
num_of_class = 5
def generate_training_images(cur_images, cur_tags, batch_size=500):

    while True:

        cur_batch = []

        cur_labels = []

        cur_wts = []

        for ix,image in enumerate(cur_images):

            #print (ix,len(cur_labels))

            label = cur_tags[ix]

            wt = wts[label]

            img = cv2.imread(image)

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            img = adjust_gamma(img, 1.5)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = add_contrast(img, 20)

            #img = preproces_image(img)

            #img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))



            new_img = deepcopy(img)

            #new_img += cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY) 

            cur_batch.append(new_img)

            cur_labels.append(label)

            cur_wts.append(wt)

            if len(cur_batch) == batch_size:

                batch_imgs = np.stack(cur_batch, axis=0)

                batch_targets = keras.utils.np_utils.to_categorical(cur_labels, num_of_class )

                yield batch_imgs,batch_targets, np.array(cur_wts)

                cur_batch = []

                cur_labels = []

                cur_wts = []





            new_img = adjust_gamma(deepcopy(img), random.uniform(0.8, 1.8))

            #new_img += cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY) 

            cur_batch.append(new_img)

            cur_labels.append(label)

            cur_wts.append(wt)

            if len(cur_batch) == batch_size:

                batch_imgs = np.stack(cur_batch, axis=0)

                batch_targets = keras.utils.np_utils.to_categorical(cur_labels, num_of_class)

                yield batch_imgs,batch_targets, np.array(cur_wts)

                cur_batch = []

                cur_labels = []

                cur_wts = []



            new_img = horizontal_flip(deepcopy(img))

            #new_img += cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY) 

            cur_batch.append(new_img)

            cur_labels.append(label)

            cur_wts.append(wt)

            if len(cur_batch) == batch_size:

                batch_imgs = np.stack(cur_batch, axis=0)

                batch_targets = keras.utils.np_utils.to_categorical(cur_labels, num_of_class)

                yield batch_imgs,batch_targets, np.array(cur_wts)

                cur_batch = []

                cur_labels = []

                cur_wts = []

        batch_imgs = np.stack(cur_batch, axis=0)

        batch_targets = keras.utils.np_utils.to_categorical(cur_labels, num_of_class )

        yield batch_imgs,batch_targets,  np.array(cur_wts)
def generate_testing_images(cur_images, cur_tags, batch_size=500):

    while True:

        cur_batch = []

        cur_labels = []

        cur_wts = []

        for ix,image in enumerate(cur_images):

            #print (ix,len(cur_labels))

            label = cur_tags[ix]

            wt = wts[label]

            img = cv2.imread(image)

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            img = adjust_gamma(img, 1.5)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = add_contrast(img, 20)

            #img = preproces_image(img)

            #img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))



            new_img = deepcopy(img)

            #new_img += cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY) 

            cur_batch.append(new_img)

            cur_labels.append(label)

            cur_wts.append(wt)

            if len(cur_batch) == batch_size:

                batch_imgs = np.stack(cur_batch, axis=0)

                batch_targets = keras.utils.np_utils.to_categorical(cur_labels, num_of_class )

                yield batch_imgs,batch_targets

                cur_batch = []

                cur_labels = []

                cur_wts = []





            new_img = adjust_gamma(deepcopy(img), random.uniform(0.8, 1.8))

            #new_img += cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY) 

            cur_batch.append(new_img)

            cur_labels.append(label)

            cur_wts.append(wt)

            if len(cur_batch) == batch_size:

                batch_imgs = np.stack(cur_batch, axis=0)

                batch_targets = keras.utils.np_utils.to_categorical(cur_labels, num_of_class)

                yield batch_imgs,batch_targets

                cur_batch = []

                cur_labels = []

                cur_wts = []



            new_img = horizontal_flip(deepcopy(img))

            #new_img += cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY) 

            cur_batch.append(new_img)

            cur_labels.append(label)

            cur_wts.append(wt)

            if len(cur_batch) == batch_size:

                batch_imgs = np.stack(cur_batch, axis=0)

                batch_targets = keras.utils.np_utils.to_categorical(cur_labels, num_of_class)

                yield batch_imgs,batch_targets

                cur_batch = []

                cur_labels = []

                cur_wts = []

        batch_imgs = np.stack(cur_batch, axis=0)

        batch_targets = keras.utils.np_utils.to_categorical(cur_labels, num_of_class )

        yield batch_imgs,batch_targets
print (len(train_images), len(train_labels))

train_labels[:1]
for batch in generate_training_images(train_images, train_labels, 100):

    tmp_images, labels, wts  = batch

    print (tmp_images[0].shape, len(labels) )

    plt.imshow(tmp_images[10])

    break
from keras.applications.densenet import DenseNet121

from keras.layers import Input

from keras.models import Model

from keras.layers import Dense

from keras.optimizers import Adam

from keras.models import load_model

from IPython.display import clear_output
def reset_tf_session():

    curr_session = tf.get_default_session()

    # close current session

    if curr_session is not None:

        curr_session.close()

    # reset graph

    K.clear_session()

    # create new session

    config = tf.ConfigProto()

    config.gpu_options.allow_growth = True

    s = tf.InteractiveSession(config=config)

    K.set_session(s)

    return s
input_shape = (IMG_SIZE, IMG_SIZE, 3)
s = reset_tf_session()
# img_in = Input(input_shape)              #input of model 

# model = DenseNet121(include_top= False , # remove  the 3 fully-connected layers at the top of the network

#                 weights='imagenet',      # pre train weight 

#                 input_tensor= img_in, 

#                 input_shape= input_shape,

#                 pooling ='avg') 



# x = model.output  

# predictions = Dense(num_of_class, activation="sigmoid", name="predictions")(x)    # fuly connected layer for predict class 

# model = Model(inputs=img_in, outputs=predictions)
#model.summary()
INIT_LR = 5e-3  # initial learning rate

BATCH_SIZE = 200

EPOCHS = 200

def lr_scheduler(epoch):

    return min(INIT_LR * 0.9 ** epoch, 0.00001)



# callback for printing of actual learning rate used by optimizer

class LrHistory(keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs={}):

        print("Learning rate:", K.get_value(model.optimizer.lr))



len(test_images)

len(test_labels)
class PlotLearning(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):

        self.i = 0

        self.x = []

        self.losses = []

        self.val_losses = []

        self.acc = []

        self.val_acc = []

        self.fig = plt.figure()

        

        self.logs = []



    def on_epoch_end(self, epoch, logs={}):

        if epoch%10 == 0 and epoch > 0:

            self.logs.append(logs)

            self.x.append(self.i)

            self.losses.append(logs.get('loss'))

            self.val_losses.append(logs.get('val_loss'))

            self.acc.append(logs.get('acc'))

            self.val_acc.append(logs.get('val_acc'))

            self.i += 1

            f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)



            clear_output(wait=True)



            ax1.set_yscale('log')

            ax1.plot(self.x, self.losses, label="loss")

            ax1.plot(self.x, self.val_losses, label="val_loss")

            ax1.legend()



            ax2.plot(self.x, self.acc, label="accuracy")

            ax2.plot(self.x, self.val_acc, label="validation accuracy")

            ax2.legend()



            plt.show();

        

plot_losses = PlotLearning()
# model.compile(

#     loss='categorical_crossentropy',  # we train 10-way classification

#     optimizer=keras.optimizers.adamax(lr=INIT_LR),  # for SGD

#     metrics=['accuracy']  # report accuracy during training

# )





#model = load_model('../input/aptos/iris_trained_model_v2')
!ls ../input/
model = None

import os

model = load_model('../input/aptosv4/iris_trained_model_v3')

print ("loaded existing model weights")
# prepare model for fitting (loss, optimizer, etc)



# model.fit_generator(epochs=EPOCHS,

#                     generator=generate_training_images(train_images, train_labels, BATCH_SIZE),

#                     steps_per_epoch = len(train_images) // BATCH_SIZE // 8,

#                      validation_steps = 40,

#                     callbacks=[keras.callbacks.LearningRateScheduler(lr_scheduler), 

#                                LrHistory(),

#                                plot_losses],

#                     validation_data=generate_testing_images(test_images, test_labels, BATCH_SIZE),

#                     use_multiprocessing=True,

#                     workers = 4,

#                     initial_epoch=0)
#model.save("./iris_trained_model_v3")
#!ls ../input/aptos2019-blindness-detection
predict_images = [f for f in glob.glob("../input/aptos2019-blindness-detection/test_images/" + "*.png")]

#labels = [ labels_df.loc[f.split('/')[-1].split('.')[0].strip()].diagnosis for f in  images]
len(predict_images)
def generate_predict_images(cur_images):

    cur_batch = []

    cur_labels = []

    cur_wts = []

    for ix,image in enumerate(cur_images):

        img = cv2.imread(image)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        img = adjust_gamma(img, 1.5)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = add_contrast(img, 20)

        cur_batch.append(img)

        cur_labels.append(image.split('/')[-1].split('.')[0])

    return cur_batch, cur_labels
pred_batch, names = generate_predict_images(predict_images)
predictions = model.predict(np.array(pred_batch) )
predictions[:5]
fl = open('submission.csv', 'w')

fl.write("id_code,diagnosis\n")

for ix, row in enumerate(predictions):

    row = list(row)

    fl.write("{},{}\n".format(str(names[ix]), str(row.index(max(row ))) ))

fl.close()
#!head -100 submission.csv
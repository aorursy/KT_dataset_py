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
filepath = '../input/dog-breed-identification/'
#problem
## identify breed of the dog , given image of dog

# evaluation
# predict probablities of each dog breed of each test image

# features
# Some information about data :
# we are dealing with unstructured data (images) , so its probably best we use deep learning / transfer learning
# 120 breeds - 120 classes
# 10000+ images in training set
# 10000+ images in test set

import tensorflow as tf
print('Tensorflow version:', tf.__version__)
import tensorflow_hub as hub
print('Tensorflow hub version:', hub.__version__)

# check for gpu availability

print("GPU available" if tf.config.list_physical_devices("GPU") else "Not available")
import pandas as pd
labelsdf = pd.read_csv(filepath+'labels.csv')
print(labelsdf.describe())
print(labelsdf.head())
labelsdf.breed.value_counts().plot.bar(figsize=(20,10));
## view an image
from IPython.display import Image
Image(filepath+'train/09839ef1c5a5a5b3acb61c4093cab07f.jpg')
## Getting images and their labels

filenames = [filepath+"train/"+fname+".jpg" for fname in labelsdf['id']]
filenames
len(filenames)
## check number of file names matches with actual images files

import os
if len(os.listdir(filepath+"train/"))  == len(filenames):
    print("file names match actual")
else:
    print("doesn't match")
Image(filenames[9000])
import numpy as np
labels = labelsdf['breed']
labels = np.array(labels)
labels
len(labels)
uniquebreeds = np.unique(labels)
uniquebreeds
# convert label to boolean arr

print(labels[0])
print(labels[0]==uniquebreeds)
bool_labels = [label == uniquebreeds for label in labels]
len(bool_labels)
## turning bool labels into int

print(labels[0])
print(np.where(labels[0]==uniquebreeds))  ## index where label occurs
print(bool_labels[0].argmax()) # index where label occurs in boolean array
print(bool_labels[0].astype(int))
# create validation set

x = filenames
y = bool_labels

numimages = 1000

from sklearn.model_selection import train_test_split

xtrain, xval, ytrain, yval = train_test_split(x[:numimages], y[:numimages], test_size=0.2, random_state=42)

print(len(xtrain), len(xval), len(ytrain), len(yval))

## preprocess images
## turning images into tensors

#convert image to numpy array
from matplotlib.pyplot import imread
image = imread(filenames[42])
print(image.shape)
tf.constant(image)
imgsize=224
def processimage(imagepath, imgsize=imgsize):
    """
    Take imagepath as input and turn image into tensor
    """
    # Read in an image
    image = tf.io.read_file(imagepath)
    # turn jpeg image into numerical tensor with 3 channels (RGB)
    image = tf.image.decode_jpeg(image, channels=3)
    #convert color channels values from 0-255 to 0-1 values
    image = tf.image.convert_image_dtype(image, tf.float32)
    # resize the image to desired value (224, 224)
    image = tf.image.resize(image, size=[imgsize, imgsize])
    
    return image
## turning data into batches
## we use batch size=32
## tensor tuples (image, label)

def get_image_label(image_path, label):
    image = processimage(image_path)
    return image, label
# turn data into batches
batchsize=32

def create_data_batches(x, y=None, batchsize=batchsize, valid_data=False, test_data=False):
    
    ## shuffles training data, but does not shuffle validation data
    
    if test_data:
        print("creating test data batches :")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x)))
        databatch = data.map(processimage).batch(batchsize)
        return databatch
    elif valid_data:
        print("creating validation data batches:")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x), tf.constant(y)))
        databatch = data.map(get_image_label).batch(batchsize)
        return databatch
    else:
        print("creating train data batches:")
        #convert to tensors
        data=tf.data.Dataset.from_tensor_slices((tf.constant(x), tf.constant(y)))
        
        #shuffling
        data=data.shuffle(buffer_size=len(x))
        data = data.map(get_image_label)
        
        #batches
        databatch=data.batch(batchsize)
        return databatch
    
        
train_data = create_data_batches(xtrain, ytrain)
val_data = create_data_batches(xval, yval, valid_data=True)

print(train_data.element_spec) 
print(val_data.element_spec)
## data visualization
import matplotlib.pyplot as plt
def show25images(images, labels):
    plt.figure(figsize=(10,10))
    for i in range(25):
        #create subplots
        ax=plt.subplot(5, 5, i+1)
        plt.imshow(images[i])
        plt.title(uniquebreeds[labels[i].argmax()])
        plt.axis('off')
        
train_images, train_labels = next(train_data.as_numpy_iterator())
len(train_images), len(train_labels)
uniquebreeds[y[4].argmax()]
show25images(train_images, train_labels)
val_images, val_labels = next(val_data.as_numpy_iterator())
show25images(val_images, val_labels)
# define inputs and outputs for the model

imgsize=224

inputshape = [None, imgsize, imgsize, 3]  # batch, height, width, color channels
outputshape = len(uniquebreeds)

## setup model url from tensorflow hub

model_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"
## keral model

def create_model(inputshape=inputshape, outputshape=outputshape, modelurl=model_url):
    print("Building model with : ", modelurl)
    
    model=tf.keras.Sequential([hub.KerasLayer(modelurl), 
                               tf.keras.layers.Dense(units=outputshape, activation='softmax')
                              ])
    
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                 optimizer=tf.keras.optimizers.Adam(),
                 metrics=['accuracy'])
    
    model.build(inputshape)
    
    return model


model = create_model()
model.summary()
## callbacks

#load tensorboard

%load_ext tensorboard


import datetime

# function to build tensor board callback

def create_tensorboard_callback():
    ## create log dir for tensorboard logs
    logdir=os.path.join("/kaggle/working/logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    return tf.keras.callbacks.TensorBoard(logdir)

## early stopping callback

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3)
numepochs=100 
def train_model():
    model = create_model()
    tensorboard = create_tensorboard_callback()
    model.fit(x=train_data, epochs=numepochs, validation_data=val_data, validation_freq=1, 
              callbacks=[tensorboard, early_stopping])
    return model
model = train_model()
## prevent overfitting

## tensorboard magic function

%tensorboard --logdir /kaggle/working/logs
predictions = model.predict(val_data, verbose=1)
predictions
predictions.shape
np.sum(predictions[0]) , np.sum(predictions[1])
np.max(predictions[0]),  np.max(predictions[1])
index=42
print(f"max :", np.max(predictions[index]))
print(f"sum :", np.sum(predictions[index]))
print(f"max index: ", np.argmax(predictions[index]))
print(f"prediction label : ", uniquebreeds[np.argmax(predictions[index])])
def get_prediction_label(predprob):
    return uniquebreeds[np.argmax(predprob)]

pred_label=get_prediction_label(predictions[9])
pred_label
val_data
## unbatch dataset

def unbatchify(data):
    images_ = []
    labels_ = []

    for image, label in data.unbatch().as_numpy_iterator():
        images_.append(image)
        labels_.append(uniquebreeds[np.argmax(label)])
    return images_, labels_
valimg, vallab = unbatchify(val_data)
get_prediction_label(vallab[0])
def plotpred(predprob, labels, images, n=1):
    pred_prob, truelabel, image = predprob[n], labels[n], images[n]
    predlabel = get_prediction_label(pred_prob)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    if (predlabel == truelabel):
        color="green"
    else:
        color="red"
        
    plt.title(f"{predlabel} {np.max(pred_prob)*100:2.0f}% {truelabel}" , color=color)
plotpred(predprob=predictions, labels=vallab, images=valimg, n=1)
def plotpredconf(predprob, labels, n=1):
    pred_prob , truelabel = predprob[n], labels[n]
    pred_label = get_prediction_label(pred_prob)
    # top 10 prediction indexes
    top_10_pred_ind = pred_prob.argsort()[-10:][::-1]
    top_10_pred_values = pred_prob[top_10_pred_ind]
    top10predlabels = uniquebreeds[top_10_pred_ind]
    
    topplot = plt.bar(np.arange(len(top10predlabels)), top_10_pred_values, color="grey")
    plt.xticks(np.arange(len(top10predlabels)), labels=top10predlabels, rotation='vertical')
    
    if np.isin(truelabel,top10predlabels):
        print("yes")
        topplot[np.argmax(top10predlabels==truelabel)].set_color('green')
    else:
        pass
    
    
plotpredconf(predictions, vallab,99)
predictions[0][predictions[0].argsort()[-10:][::-1]]
predictions[0].max()
imult=0
rows=3
cols=2
numimg = rows * cols
plt.figure(figsize=(10*cols, 5*rows))
for i in range(numimg):
    plt.subplot(rows, 2*cols, 2*i+1)
    plotpred(predictions, vallab, valimg, n=i+imult)
    plt.subplot(rows, 2*cols, 2*i+2)
    plotpredconf(predictions, vallab, n=i+imult)
plt.tight_layout(h_pad=1.0)
plt.show()

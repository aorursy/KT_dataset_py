# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





#Be sure to turn on GPU and set Internet to "Internet connected"





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../working"))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

tf.test.gpu_device_name()
from glob import glob

from sklearn.model_selection import train_test_split



daisy_train = glob('../input/transfer-flowers/flowers/flowers/Train/daisy/*.jpg')

dandelion_train = glob('../input/transfer-flowers/flowers/flowers/Train/dandelion/*.jpg')

rose_train = glob('../input/transfer-flowers/flowers/flowers/Train/rose/*.jpg')

sunflower_train = glob('../input/transfer-flowers/flowers/flowers/Train/sunflower/*.jpg')

tulip_train = glob('../input/transfer-flowers/flowers/flowers/Train/tulip/*.jpg')



print('daisy_train',len(daisy_train))

print('dandelion_train',len(dandelion_train))

print('rose_train',len(rose_train))

print('sunflower_train',len(sunflower_train))

print('tulip_train',len(tulip_train))



TRAIN_DIR = '../input/transfer-flowers/flowers/flowers/Train'

TEST_DIR = '../input/transfer-flowers/flowers/flowers/Test'

VALID_DIR = '../input/transfer-flowers/flowers/flowers/Valid'

import numpy as np

from PIL import Image

import matplotlib.pyplot as plt



daisy = np.random.choice(daisy_train, 5)

dandelion = np.random.choice(dandelion_train, 5)

rose = np.random.choice(rose_train, 5)

sunflower = np.random.choice(sunflower_train, 5)

tulip = np.random.choice(tulip_train, 5)

data = np.concatenate((daisy,dandelion,rose,sunflower,tulip))

labels = 5 * ['daisy'] + 5 * ['dandelion'] + 5 * ['rose'] + 5 * ['sunflower'] + 5 * ['tulip'] 



N, R, C = 25, 5, 5

plt.figure(figsize=(12, 9))

for k, (src, label) in enumerate(zip(data, labels)):

    im = Image.open(src).convert('RGB')

    plt.subplot(R, C, k+1)

    plt.title(label)

    plt.imshow(np.asarray(im))

    plt.axis('off')
from keras.models import Model

from keras.models import Sequential

from keras.layers import Dense, GlobalAveragePooling2D, Dropout

from keras.applications.inception_v3 import InceptionV3, preprocess_input



CLASSES = 5

    

# setup model

base_model = InceptionV3(weights='imagenet', include_top=False)



# Add more layers

model = Sequential()

model.add(base_model)

model.add(GlobalAveragePooling2D(name='avg_pool'))



#add more layers here if needed



model.add(Dense(CLASSES, activation='softmax'))

#model = Model(inputs=base_model.input, outputs=predictions)

   

# transfer learning - set all layers of the base model to frozen

for layer in base_model.layers:

    layer.trainable = False

      

model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['accuracy'])

base_model.summary()
for layer in base_model.layers:

    layer.trainable = False
# Unfreeze the last x layers from bottom

#x = 0

#for layer in base_model.layers[-x:]:

#     layer.trainable = True

#model.compile(optimizer='adam',

#              loss='categorical_crossentropy',

#              metrics=['accuracy'])



for layer in base_model.layers:

    if layer.trainable == True:

        print(layer.name)
from keras.preprocessing.image import ImageDataGenerator



WIDTH = 299

HEIGHT = 299

BATCH_SIZE = 128



# data prep

train_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    vertical_flip=True,

    fill_mode='nearest')



validation_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    rotation_range=0,

    width_shift_range=0,

    height_shift_range=0,

    shear_range=0,

    zoom_range=0,

    horizontal_flip=False,

    fill_mode='nearest')



train_generator = train_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=(HEIGHT, WIDTH),

		batch_size=247,

		class_mode='categorical')

    

validation_generator = validation_datagen.flow_from_directory(

    VALID_DIR,

    target_size=(HEIGHT, WIDTH),

    batch_size=429,

    class_mode='categorical')
x_batch, y_batch = next(train_generator)



plt.figure(figsize=(12, 9))

for k, (img, lbl) in enumerate(zip(x_batch[0:24], y_batch[0:24])):

    plt.subplot(4, 6, k+1)

    plt.imshow((img + 1) / 2)

    plt.axis('off')
from keras.callbacks import ModelCheckpoint

EPOCHS = 1

STEPS_PER_EPOCH = 1

VALIDATION_STEPS = 1



checkpointer = ModelCheckpoint(filepath='weights_{epoch:02d}_{val_loss:.4f}_hdf5', 

                               verbose=1, save_best_only=True)



history = model.fit_generator(

    train_generator,

    epochs=EPOCHS,

    steps_per_epoch=STEPS_PER_EPOCH,

    validation_data=validation_generator,

    validation_steps=VALIDATION_STEPS,

    callbacks=[checkpointer])

#model.load_weights('')
# code to clean up the working directory in case you fill it with checkpoints

###WARNING####. this will delete all checkpoints, adjust as needed

#import os

#files = os.listdir('../working')

#for file in files:

#    if file.endswith("f5"):

#        os.remove(os.path.join('../working',file))
#plot the training history



def plot_training(history):

  acc = history.history['acc']

  val_acc = history.history['val_acc']

  loss = history.history['loss']

  val_loss = history.history['val_loss']

  epochs = range(len(acc))

  

  plt.plot(epochs, acc, 'r.')

  plt.plot(epochs, val_acc, 'r')

  plt.title('Training and validation accuracy')

  

  plt.figure()

  plt.plot(epochs, loss, 'r.')

  plt.plot(epochs, val_loss, 'r-')

  plt.title('Training and validation loss')

  plt.show()

  

plot_training(history)
#Create a single test set from all of the test directories

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec



from keras.preprocessing import image

from keras.models import load_model

from tqdm import tqdm

import cv2



X = []

X_p = []

Y = []

files = []

def make_train_data(DIR,label):

    for img in tqdm(os.listdir(DIR)):

        path = os.path.join(DIR,img)

        if path.endswith('.jpg'):

            img = image.load_img(path, target_size=(HEIGHT, WIDTH))

            img_p = Image.open(path).convert('RGB')

            x = image.img_to_array(img)

            #x = np.expand_dims(x, axis=0)

            x = preprocess_input(x)

            X.append(np.array(x))

            Y.append(label)

            X_p.append(np.array(img_p))

            files.append(path)

make_train_data('../input/transfer-flowers/flowers/flowers/Test/daisy/',0)

make_train_data('../input/transfer-flowers/flowers/flowers/Test/dandelion/',1)

make_train_data('../input/transfer-flowers/flowers/flowers/Test/rose/',2)

make_train_data('../input/transfer-flowers/flowers/flowers/Test/sunflower/',3)

make_train_data('../input/transfer-flowers/flowers/flowers/Test/tulip/',4)

print(len(X))

X=np.array(X)

X_p=np.array(X_p)
import numpy as np

from sklearn.metrics import accuracy_score



predicts = model.predict(X)

predicts = np.argmax(predicts, axis=1)



accuracy_score(Y, predicts)
#show first 25 misclassified Daisies

labels = ("daisy","dandelion","rose","sunflower","tulip")



N, R, C = 25, 5, 5

count = 0

plt.figure(figsize=(12, 9))

for k in range(len(predicts)):

    if count<25:

        if (Y[k] == 0) & (predicts[k] != 0):

            im = Image.open(files[k]).convert('RGB')

            plt.subplot(R, C, count+1)

            plt.title(labels[predicts[k]])

            plt.imshow(np.asarray(im))

            plt.axis('off')

            count = count + 1

    
#misclassified Dandelion

N, R, C = 25, 5, 5

count = 0

plt.figure(figsize=(12, 9))

for k in range(len(predicts)):

    if count<25:

        if (Y[k] == 1) & (predicts[k] != 1):

            im = Image.open(files[k]).convert('RGB')

            plt.subplot(R, C, count+1)

            plt.title(labels[predicts[k]])

            plt.imshow(np.asarray(im))

            plt.axis('off')

            count = count + 1

    
#misclassified Roses

N, R, C = 25, 5, 5

count = 0

plt.figure(figsize=(12, 9))

for k in range(len(predicts)):

    if count<25:

        if (Y[k] == 2) & (predicts[k] != 2):

            im = Image.open(files[k]).convert('RGB')

            plt.subplot(R, C, count+1)

            plt.title(labels[predicts[k]])

            plt.imshow(np.asarray(im))

            plt.axis('off')

            count = count + 1
#misclassified Sunflowers

N, R, C = 25, 5, 5

count = 0

plt.figure(figsize=(12, 9))

for k in range(len(predicts)):

    if count<25:

        if (Y[k] == 3) & (predicts[k] != 3):

            im = Image.open(files[k]).convert('RGB')

            plt.subplot(R, C, count+1)

            plt.title(labels[predicts[k]])

            plt.imshow(np.asarray(im))

            plt.axis('off')

            count = count + 1
#misclassified Tulips

N, R, C = 25, 5, 5

count = 0

plt.figure(figsize=(12, 9))

for k in range(len(predicts)):

    if count<25:

        if (Y[k] == 4) & (predicts[k] != 4):

            im = Image.open(files[k]).convert('RGB')

            plt.subplot(R, C, count+1)

            plt.title(labels[predicts[k]])

            plt.imshow(np.asarray(im))

            plt.axis('off')

            count = count + 1
#Function for plotting confusion matrix



import itertools

from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()
#Plot confusion matrix

cnf_matrix = confusion_matrix(Y, predicts)

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plot_confusion_matrix(cnf_matrix, classes=labels,

                      title='Confusion matrix, without normalization')



plt.show()


for i in range(len(Y)):

    if (predicts[i] == 2) & (Y[i] == 2):

        print(i)
#a few functions to create a single image view



import numpy as np

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec



from keras.preprocessing import image

from keras.models import load_model





def predict(model, img):

    """Run model prediction on image

    Args:

        model: keras model

        img: PIL format image

    Returns:

        list of predicted labels and their probabilities 

    """

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    preds = model.predict(x)

    return preds[0]





def plot_preds(img, preds,Y):

    """Displays image and the top-n predicted probabilities in a bar graph

    Args:

        preds: list of predicted labels and their probabilities

    """

    labels = ("daisy","dandelion","rose","sunflower","tulip")

    preds_pos = np.arange(len(labels))

    preds_label = labels[Y]

    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

    plt.figure(figsize=(8,8))

    plt.subplot(gs[0])

    plt.imshow(np.asarray(img))

    plt.subplot(gs[1])

    plt.barh(preds_pos, preds, alpha=0.5)

    plt.yticks(preds_pos, labels)

    plt.xlabel('Probability')

    plt.xlim(0, 1)

    plt.tight_layout()

    return preds_label
img_num = 248

img = image.load_img(files[img_num], target_size=(HEIGHT, WIDTH))

preds = predict(model, img)

preds_label = plot_preds(np.asarray(img), preds,Y[img_num])

preds_label
#creates a list of numbers from 0 to length of images for the ImageID field

list_of_num = [i for i in range(0,len(predicts))]



#Creates a dataframe that can be saved as a csv for submission

submission_data = pd.DataFrame(

    {'ImageId': list_of_num,

     'Label': predicts

    })
# code to allow you to download submission csv with commiting

# https://www.kaggle.com/rtatman/download-a-csv-file-from-a-kernel

# import the modules we'll need

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a link to download the dataframe

create_download_link(submission_data)

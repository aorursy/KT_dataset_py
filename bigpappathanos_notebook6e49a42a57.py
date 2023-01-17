# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# uncomment the following code to see all files

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import plotly.express as px

import plotly.graph_objects as go

import plotly.offline as po

from PIL import Image

from random import randint, shuffle

po.init_notebook_mode(connected = True)
# this is how the dataset is by default. 

# note that I am using the val data as test data and the test data as validation data



train_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/train/'

val_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/test/'

test_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/val/'



segments = [train_dir + 'NORMAL', train_dir + 'PNEUMONIA',test_dir + 'NORMAL', test_dir + 'PNEUMONIA',val_dir + 'NORMAL', val_dir + 'PNEUMONIA']

labels = ['train_normal', 'train_pneumonia','test_normal', 'test_pneumonia','val_normal', 'val_pneumonia',]



segments = [len(os.listdir(item)) for item in segments]

fig = px.pie(values = segments, 

             names = labels,  

             color_discrete_sequence=px.colors.sequential.RdBu,

             width = 450, height = 450)



fig.show()
from plotly.subplots import make_subplots



fig = make_subplots(rows = 2, cols = 4, subplot_titles = ['PNEUMONIA','PNEUMONIA','PNEUMONIA','PNEUMONIA','NORMAL','NORMAL','NORMAL','NORMAL'])

# plotting 8 random images from the train directory 



def generate_random_image(case):

    cases = [ '/PNEUMONIA/','/NORMAL/' ]



    name = train_dir + cases[case] + os.listdir(train_dir + cases[case])[randint(0, len(os.listdir(train_dir + cases[case]))- 1)]

    image = Image.open(name).resize((150, 150), Image.ANTIALIAS).transpose(Image.FLIP_TOP_BOTTOM)

    image_arr = np.asarray(image)

    return [image_arr, cases[case]]





    





fig.add_trace(px.imshow(generate_random_image(1)[0], color_continuous_scale = 'Viridis', title = 'dfsdf')['data'][0], row = 1, col = 1)

fig.add_trace(px.imshow(generate_random_image(1)[0], color_continuous_scale = 'Viridis')['data'][0], row = 1, col = 2)

fig.add_trace(px.imshow(generate_random_image(1)[0], color_continuous_scale = 'Viridis')['data'][0], row = 1, col = 3)

fig.add_trace(px.imshow(generate_random_image(1)[0], color_continuous_scale = 'Viridis')['data'][0], row = 1, col = 4)

fig.add_trace(px.imshow(generate_random_image(0)[0], color_continuous_scale = 'Viridis')['data'][0], row = 2, col = 1)

fig.add_trace(px.imshow(generate_random_image(0)[0], color_continuous_scale = 'Viridis')['data'][0], row = 2, col = 2)

fig.add_trace(px.imshow(generate_random_image(0)[0], color_continuous_scale = 'Viridis')['data'][0], row = 2, col = 3)

fig.add_trace(px.imshow(generate_random_image(0)[0], color_continuous_scale = 'Viridis')['data'][0], row = 2, col = 4)





fig.update_layout(height = 1000, width = 1500)

fig.show()
IMG_SIZE = 300



train_images_normal = os.listdir(train_dir + 'NORMAL')

train_images_normal = [train_dir + 'NORMAL/' + item for item in train_images_normal]



train_images_pneu = os.listdir(train_dir + 'PNEUMONIA')

train_images_pneu = [train_dir + 'PNEUMONIA/' + item for item in train_images_pneu]



test_images_normal = os.listdir(test_dir + 'NORMAL')

test_images_normal = [test_dir + 'NORMAL/' + item for item in test_images_normal]



test_images_pneu = os.listdir(test_dir + 'PNEUMONIA')

test_images_pneu = [test_dir + 'PNEUMONIA/' + item for item in test_images_pneu]



train_images = train_images_normal + train_images_pneu

train_labels = [0]*len(train_images_normal) + [1]*len(train_images_pneu)

test_images = test_images_normal + test_images_pneu

test_labels = [0]*len(test_images_normal) + [1]*len(test_images_pneu)





z1 = list(zip(train_images, train_labels))

z2 = list(zip(test_images, test_labels))



# randomly shuffling images

shuffle(z1)

shuffle(z2)



train_images, train_labels = zip(*z1)

test_images, test_labels = zip(*z2)







from tqdm import tqdm



def img_to_num(img_lst, labels, case):

    tmp, tmp2 = [], []

    

    for img_name, label in tqdm(zip(img_lst, labels)):

        # this code was showing the image upside down to I flipped it vertically again.

        # also I resized the image to 300, 300 px. and converted them to grayscale.

        image = Image.open(img_name).convert('LA').resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS).transpose(Image.FLIP_TOP_BOTTOM)

        # Converting image to numbers

        image_arr = np.asarray(image)

        tmp.append(image_arr)

        tmp2.append(label)

        

        # I aLso added the mirror image of each image to the dataset. 

        if case == 'train':

            tmp.append(np.asarray(image.transpose(Image.FLIP_LEFT_RIGHT)))

            tmp2.append(label)

    return tmp, tmp2





train_images, train_labels, = img_to_num(list(train_images), list(train_labels), 'train')

test_images, test_labels = img_to_num(list(test_images), list(test_labels), 'test')

print(train_images[0:3])

print()

print(train_labels[0:3])
# processing validation data...



val_images_normal = os.listdir(val_dir + 'NORMAL')

val_images_normal = [val_dir + 'NORMAL/' + item for item in val_images_normal]



val_images_pneu = os.listdir(val_dir + 'PNEUMONIA')

val_images_pneu = [val_dir + 'PNEUMONIA/' + item for item in val_images_pneu]



val_images = val_images_normal + val_images_pneu

val_labels = [0]*len(val_images_normal) + [1]*len(val_images_pneu)







from tqdm import tqdm



def img_to_num(img_lst, labels):

    tmp, tmp2 = [], []

    

    for img_name, label in tqdm(zip(img_lst, labels)):

        image = Image.open(img_name).convert('LA').resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS).transpose(Image.FLIP_TOP_BOTTOM)

        image_arr = np.asarray(image)

        tmp.append(image_arr)

        tmp2.append(label)

    return tmp, tmp2





val_images, val_labels, = img_to_num(list(val_images), list(val_labels))





segments = [len(train_images), len(test_images), len(val_images)]

labels = ['Training Data', 'Test Data', 'Validation Data']

for a,b in zip(segments, labels):

    print(a,b)



fig = px.pie(values = segments, 

             names = labels,  

             color_discrete_sequence=px.colors.sequential.RdBu,

             width = 450, height = 450)



fig.show()
print(len(train_images), len(train_labels))

shapes = [item.shape for item in tqdm(train_images)]

from collections import Counter

print(Counter(shapes).keys())

print(Counter(shapes).values())
# let's build the model now!!!



import tensorflow as tf





model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, kernel_size = (3,3), activation = 'relu', padding = 'same', input_shape = (300, 300, 2)),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.SeparableConv2D(64, kernel_size = (3,3), activation = 'relu', padding = 'same'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.SeparableConv2D(256, kernel_size = (3,3), activation = 'relu', padding = 'same'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(512, kernel_size = (3,3), activation = 'relu', padding = 'same'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(512, kernel_size = (3,3), activation = 'relu', padding = 'same'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(4096, activation = 'relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(2048, activation = 'relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(1024, activation = 'relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(512, activation = 'relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(256, activation = 'relu'),

    tf.keras.layers.Dense(128, activation = 'relu'),

    tf.keras.layers.Dense(1, activation = 'sigmoid')

])



opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer =opt, loss = 'binary_crossentropy', metrics = ['accuracy'])

model.summary()



history = model.fit(np.array(train_images),np.array(train_labels),

                     epochs = 10, 

                      verbose = 2, validation_data = (np.array(val_images), np.array(val_labels)))
# testing the model on test images (16 images)

model.evaluate(np.array(test_images), np.array(test_labels), verbose = 2 )

predictions = model.predict(np.array(test_images))

actual = np.array(test_labels)



lst = []

for pred, act in zip(predictions, actual):

    if pred == act:

        lst.append('predicted correctly')

    else:

        lst.append('predicted incorrectly')

        

px.histogram(lst).show()
fig = go.Figure()

fig.add_trace(go.Scatter(y = history.history['accuracy'], x = list(range(0, 30)), name = 'accuracy'))

fig.add_trace(go.Scatter(y = history.history['val_accuracy'], x = list(range(0, 30)), name = 'val_accuracy'))

fig.show()



fig= go.Figure()

fig.add_trace(go.Scatter(y = history.history['val_loss'], x = list(range(0, 30)),name = 'val_loss'))

fig.add_trace(go.Scatter(y = history.history['loss'], x = list(range(0, 30)), name = 'loss'))

fig.show()
model.save('test.h5') 
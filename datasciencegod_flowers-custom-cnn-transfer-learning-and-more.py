# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random as rand

from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle



import matplotlib.pyplot as plt # Plotting

import seaborn as sns # Plotting

from IPython.display import Image, display # Display images

import cv2 # Opencv for computer vision tasks



from keras.models import Sequential, Model, load_model  # for building sequential model

from keras.layers import Conv2D, Dense, Dropout, Activation, MaxPooling2D, Flatten, BatchNormalization

from keras.layers.advanced_activations import LeakyReLU



from keras.applications import InceptionV3, ResNet50, VGG16

from keras.applications.vgg16 import preprocess_input, decode_predictions

from keras.applications.inception_v3 import preprocess_input, decode_predictions



from keras.preprocessing.image import ImageDataGenerator



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

from tqdm import tqdm # Visualize loop progress

print(os.listdir("../input"))



# Ignore  the warnings

import warnings

warnings.filterwarnings('always')

warnings.filterwarnings('ignore')



# Any results you write to the current directory are saved as output.
print(os.listdir("../input/flowers/flowers/"))
flower_path = "../input/flowers/flowers/"

flowers_dict = {}



for flower in os.listdir(flower_path):

    folder_path = os.path.join(flower_path, flower)

    flowers = os.listdir(folder_path)

    

    flowers_dict[flower] = [folder_path, flowers]

    img_idx = rand.randint(0,len(flowers)-1)

    flwr_img_path = os.path.join(flower_path, flower, flowers[img_idx])

    print(flwr_img_path)

    print('Name of Flower: ', flower, u'\u2193', 'at index: ', img_idx)

    display(Image(filename=flwr_img_path))
def get_unique_image_shapes(imgs, path):

    shape_set = set()

    for img in imgs:

        img_path = os.path.join(path, img)

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)

        

        if image is not None:

            shape_set.add(image.shape)

    return shape_set



def get_shape_ranges(shapes):

    shapes = list(shapes)

    

    min_row = min(shapes, key=lambda item: item[0])[0]

    max_row = max(shapes, key=lambda item: item[0])[0]

    

    row_range = (min_row, max_row)

    

    min_col = min(shapes, key=lambda item: item[1])[1]

    max_col = max(shapes, key=lambda item: item[1])[1]

    

    col_range = (min_col, max_col)



    return row_range, col_range



def plot_shape_hist(shapes):

    rows = [s[0] for s in shapes]

    cols = [s[1] for s in shapes]

    

    sns.distplot(rows, label='rows')

    sns.distplot(cols, label='col')

    

    rowhist, rowbins = np.histogram(rows, bins =50)

    colhist, colbins = np.histogram(cols, bins =50)

    print("Most common row size: ",round(rowbins[np.argmax(rowhist)]))

    print("Most common col size: ",round(colbins[np.argmax(colhist)]))

    plt.legend()

    

    plt.show()

    

for i in flowers_dict:

    path = flowers_dict[i][0]

    imgs = flowers_dict[i][1]

    shapes = get_unique_image_shapes(imgs, path)

    row_range, col_range = get_shape_ranges(shapes)

    print('Row range for {} images: {}'.format(i, row_range))

    print('Column range for {} images: {}'.format(i, col_range))

    

    plot_shape_hist(shapes)
IMG_ROW = 150

IMG_COL = 150

NUM_CLASSES = 5
parent_dir = "../input/flowers/flowers/"

def prepare_datadf(parent_dir):

    df = pd.DataFrame(columns = ['path', 'label'])

    

    for flower in os.listdir(parent_dir):

        folder_path = os.path.join(parent_dir, flower)

        flowers = os.listdir(folder_path)

        for i in flowers:

            df = df.append(pd.DataFrame({'path':[os.path.join(flower, i)], 'label':[flower]}), 

                           ignore_index=True)

    

    # Shuffling for randomness

    df = df.sample(frac=1.0).reset_index(drop=True)

    return df
# Create a dataframe with paths and labels

datadf = prepare_datadf(parent_dir)

train, test = train_test_split(datadf, test_size=0.053)

val, test = train_test_split(test, test_size=0.04)
# Creating training and validation generators with data augmentation in train generator

gen = ImageDataGenerator(rotation_range=10, # in degrees 0-180

                        zoom_range=0.1, # 10% zoom

                        width_shift_range = 0.1, # 10% of horizontal shift

                        height_shift_range = 0.1, # 10% vertical shift

                        horizontal_flip = True, # flip horizontally

                        shear_range = 0.1, # 10% shear

                        rescale = 1./255) # bring all pixel values between 0 and 1.



valgen = ImageDataGenerator(rescale = 1./255)
# Creating train and validation generator instances to read image paths from dataframe 

train_generator=gen.flow_from_dataframe(dataframe=train, 

                                        directory=parent_dir, x_col="path", y_col="label", 

                                        class_mode="categorical", target_size=(IMG_ROW,IMG_COL), 

                                        batch_size=256)

val_generator = valgen.flow_from_dataframe(dataframe=val,

                                          directory=parent_dir,x_col="path", y_col="label", 

                                          class_mode="categorical", target_size=(IMG_ROW,IMG_COL), 

                                           batch_size=64)
# Defining our custom cnn model

def cnn_model():

    model = Sequential()

    

    # First Conv layer

    model.add(Conv2D(filters=32,kernel_size=(5,5), padding='same', input_shape=(IMG_ROW, IMG_COL, 3)))

    model.add(LeakyReLU(alpha=0.02)) # Activation layer

    model.add(MaxPooling2D(pool_size=(2,2))) # Pooling layer

    

    # Second Conv layer

    model.add(Conv2D(filters=196, kernel_size=(5,5)))

    model.add(LeakyReLU(alpha=0.02))

    model.add(MaxPooling2D(pool_size=(2,2)))

    

    # Third Conv layer

    model.add(Conv2D(filters=256, kernel_size=(5,5)))

    model.add(LeakyReLU(alpha=0.02))

    model.add(MaxPooling2D(pool_size=(2,2)))



    # Forth Conv layer

    model.add(Conv2D(filters=512, kernel_size=(5,5)))

    model.add(LeakyReLU(alpha=0.02))

    model.add(MaxPooling2D(pool_size=(2,2)))

    

    # Flatten

    model.add(Flatten())

    

    # Fully connected layer 1

    model.add(Dense(1024))

    model.add(LeakyReLU(alpha=0.02))

    

    # Fully connected layer 2

    model.add(Dense(512))

    model.add(LeakyReLU(alpha=0.02))

    

    # Output Layer

    model.add(Dense(NUM_CLASSES, activation='softmax'))

    

    # Compile the model

    model.compile(optimizer = 'adam', loss ='categorical_crossentropy', metrics = ['accuracy'])

    

    return model
# Define training params

batch_size = 400

epochs = 1 # Set to 1 for demo, for good results set 100, Running for 100 epochs will take time.

train_steps_per_epoch = train_generator.n//train_generator.batch_size

val_steps_per_epoch = val_generator.n//val_generator.batch_size
# Train the model

model = cnn_model()

model.fit_generator(generator=train_generator,

                    steps_per_epoch=train_steps_per_epoch,

                    validation_data=val_generator,

                    validation_steps=val_steps_per_epoch,

                    epochs=epochs)
## epoch set 1 for demo, set 100 for good results
# Inception Net

def inception():

    # Build a Sequential Model

    model = Sequential()

    

    # Add Inception module

    base_model = InceptionV3(include_top=False, input_shape=(IMG_ROW, IMG_COL, 3))

    

    ### Freezing initial layers

    for layer in base_model.layers[:249]:

        layer.trainable = False

    

    # Setting last layers as trainable | Unfreezing later layers

    for layer in base_model.layers[249:]:

        layer.trainable = True



    model.add(base_model)

    

    # Flatten *** Most Important *** Never forget to flatten a conv output before dense

    ## This is necessary for resolving dimension errors 

    model.add(Flatten()) 

    

    # Fully connected layer 1

    model.add(Dense(1024))

    model.add(LeakyReLU(alpha=0.02))

    model.add(Dropout(rate=0.1))

    

    # Fully connected layer 2

    model.add(Dense(512))

    model.add(LeakyReLU(alpha=0.02))

    

    # Output Layer

    model.add(Dense(NUM_CLASSES, activation='softmax'))

    

    # Compile the model

    model.compile(optimizer = 'adam', loss ='categorical_crossentropy', metrics = ['accuracy'])

    

    return model
# Train the model

incept_mod1 = inception()

incept_mod1.fit_generator(generator=train_generator,

                    steps_per_epoch=train_steps_per_epoch,

                    validation_data=val_generator,

                    validation_steps=val_steps_per_epoch,

                    epochs=epochs)
# Resnet

def resnet():

    # Build a Sequential Model

    model = Sequential()

    

    # Add Inception module

    model.add(ResNet50(include_top=False, input_shape=(IMG_ROW, IMG_COL, 3)))

    

    # Flatten *** Most Important *** Never forget to flatten a conv output before dense

    ## This is necessary for resolving dimension errors 

    model.add(Flatten()) 

    

    # Fully connected layer

    model.add(Dense(1024))

    model.add(LeakyReLU(alpha=0.02))

    

    # Fully connected layer

    model.add(Dense(512))

    model.add(LeakyReLU(alpha=0.02))

    

    # Output Layer

    model.add(Dense(NUM_CLASSES, activation='softmax'))

    

    # Set trainable false to use pretrained weights and not update them

    model.layers[0].trainable = False

    

    # Compile the model

    model.compile(optimizer = 'adam', loss ='categorical_crossentropy', metrics = ['accuracy'])

    

    return model
# Train the model

resmod1 = resnet()

resmod1.fit_generator(generator=train_generator,

                    steps_per_epoch=train_steps_per_epoch,

                    validation_data=val_generator,

                    validation_steps=val_steps_per_epoch,

                    epochs=epochs)
def get_intermediate_activations(img):

    

    # Read image from path

    image = cv2.imread(img, cv2.IMREAD_COLOR)

    plt.imshow(image)

    

    # Preprocess the image

    img_resized = cv2.resize(image, (299, 299))

    x = np.expand_dims(img_resized, axis=0)

    x = preprocess_input(x)

    

    # Load pretrained model

    base_model = InceptionV3(weights='imagenet')

    

    # Select layers for getting activations

    layer_outs = [layer.output for layer in base_model.layers[5:]]

    model = Model(inputs=base_model.input, outputs=layer_outs)

    

    # Make predictions

    act = model.predict(x)

    pred = base_model.predict(x)

    return act, pred

img = '../input/flowers/flowers/daisy/4511693548_20f9bd2b9c_m.jpg'

act,pred  = get_intermediate_activations(img)

decode_predictions(pred)
def display_activation(activations, col_size, row_size, act_index): 

    activation = activations[act_index]

    activation_index=0

    cnn_imgs = []

    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))

    for row in range(0,row_size):

        for col in range(0,col_size):

            cnn_imgs.append(activation[0, :, :, activation_index:activation_index+3])

            ax[row][col].imshow(((activation[0, :, :, activation_index:activation_index+3])*255).

                                astype(np.uint8))

            activation_index += 1

    return cnn_imgs
k =display_activation(act,5,5,1)    
plt.imshow((k[24]*255).astype(np.uint8))
model.save('mycnn_model.h5') # Save a model as HDF5 file



loaded_model = load_model('mycnn_model.h5') # Load the saved model for predictions
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc

import random

import os

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing import image

from tensorflow.keras.layers import *

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.applications import xception

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.model_selection import train_test_split
pd.options.display.max_rows = 100
#defining directory paths

image_dir = '/kaggle/input/analog-clocks/analog_clocks/images/'

labels_dir = '/kaggle/input/analog-clocks/analog_clocks/label.csv'
#examining labels

labels = pd.read_csv(labels_dir)

labels.describe()
#distribution of classes in hour column

labels.hour.value_counts(normalize=True)
#distribution of classes in minute column

labels.minute.value_counts(normalize=True)
#preview of dataset

sample_dir = '/kaggle/input/analog-clocks/analog_clocks/samples/'

fig = plt.figure(figsize=(20, 12))

plt.suptitle('Examples from Dataset')

for i, file in enumerate(os.listdir(sample_dir)):

    img = image.load_img(os.path.join(sample_dir, file),

                         interpolation='box')

    img = image.img_to_array(img, dtype='float32')

    img /= 255.0

    plt.subplot(2, 3, i+1)

    plt.imshow(img)

    plt.title('Sample ' + str(i+1))
#transforming labels to multi-label binary format

labels_df = pd.read_csv(labels_dir)

labels_df['tuples'] = [tuple(x) for x in labels_df.values]

labels_df['tuples'] = [('h' + str(x), 'm' + str(y)) for x,y in labels_df['tuples'].values]

labels_df = labels_df.drop(columns=['hour', 'minute'])

# labels_df = labels_df.reset_index()

binarizer = MultiLabelBinarizer()

y = binarizer.fit_transform(labels_df['tuples'])
#preview of dataframe

labels_df.head()
#distribution of unique labels

labels_df['tuples'].value_counts()
#creating train-test split

train, test = train_test_split(labels_df, 

                               stratify=labels_df['tuples'],

                               test_size=0.20,

                               random_state=42

                              )



train_idx, test_idx = list(train.index), list(test.index)
def generate(image_directory, labels, train_idx=None, batch_size=64, size=(224, 224)):

    

    """

    Function to create generator of images and labels for the neural network. This allows for training

    the model with the limited memory available. The images and labels are generated in batches of a given size.

    The images are loaded, added to a batch, preprocessed and have their features extracted using a prebuilt model

    (in this case Xception Model). 

    

    Parameters

    ----------

    image_directory: str

        The path where the images are located

    labels: array-like or list

        list of labels in multi-label binary format

    batch_size: int, default=64

        the number of images per batch

    size: tuple, default=(224, 224)

        the height and width to which the image is resized. 

    

    Yields

    ------

    image_batch: array

        Array of image features of size=batch_size

    labels_batch: array

        Array of labels in multi-label binary format of size=batch_size

    

    """

    

   

    prebuilt_model = xception.Xception(include_top=True,                      

              weights='imagenet')                                            #loading prebuilt model

    

    xception_model = Model(inputs=prebuilt_model.input,        

                           outputs=prebuilt_model.layers[-2].output)         #repurposing prebuilt model for feature extraction

    

    

    

    while 1:

        

        if train_idx==None:

            image_filenames = os.listdir(image_directory)                    #obtaining list of image filenames

        else:

            image_filenames = [str(idx) + '.jpg' for idx in train_idx]

            

        random.shuffle(image_filenames)                                      #shuffling the list to add randomness every epoch



        

        image_batch = []                                                     #initializing empty image batch list

        labels_batch = []                                                    #initializing empty labels batch list

        

        for file in image_filenames:                                         #looping over all images in directory



            index = int(file.split('.')[0])                                  #extracting image number/index from filename

            

            img = image.load_img(os.path.join(image_directory, file),        #loading image

                                 target_size=size,

                                 interpolation='box')

            

            img_arr = image.img_to_array(img, dtype='float32')               #converting image to array

            

            label = labels[index]                                            #using image number/index to find correct label in dataframe

    

            image_batch.append(img_arr)                                      #appending the image to the batch

            labels_batch.append(label)                                       #appending the label to the batch



    

            if len(image_batch)==batch_size:                                 #check to see if batch has required size

                image_batch = np.array(image_batch)                          #converting image batch list to array

                image_batch = xception.preprocess_input(image_batch)         #using xception preprocessing on image batch array

                image_features = xception_model.predict(image_batch)         #using prebuilt xception model to extract features from batch

                image_batch = np.array(image_features)                       #converting features to array

                image_batch = image_batch.reshape(batch_size,                #reshaping feature array

                                                  image_features.shape[1])   

                labels_batch = np.array(labels_batch)                        #converting labels batch list to array

                yield image_batch, labels_batch                              #yielding image and labels batch array

                image_batch = []                                             #reinitializing the image batch

                labels_batch = []                                            #reinitializing the label batch

                gc.collect()                                                 #collecting garbage to free memory

#Defining training parameters

BATCH_SIZE = 256

IMAGE_SIZE = (299, 299) #this is the size suggested for Xception model

EPOCHS = 10

STEPS = int(len(train_idx) / BATCH_SIZE)
#testing generator

sample_generator = next(generate(image_directory=image_dir, 

                                 labels=y,

                                 train_idx=train_idx,

                                 batch_size=1, 

                                 size=IMAGE_SIZE))
#output of generator

sample_generator
#extracting input and output dims from generator

INPUT_DIM = sample_generator[0][0].shape

OUTPUT_DIM = sample_generator[1].shape[1]
print(INPUT_DIM, OUTPUT_DIM)
def create_model(input_shape, output_shape):

    

    """

    Function to build and compile neural network to predict analog clocks from images

    

    Parameters

    ----------

    input_shape: tuple

        Shape tuple not including the batch_size, example: (2048, )

    output_shape: int

        Number of nodes in final layer

    

    Returns

    -------

    model: Keras model object

        A compiled Keras model

    """



    input_layer = Input(shape=input_shape)

    norm  = BatchNormalization()(input_layer)

    drop = Dropout(0.25)(norm)

    fc1 = Dense(256, activation='relu')(norm)

    fc2 = Dense(256, activation='relu')(fc1)

    output1 = Dense(output_shape, activation='sigmoid')(fc2)

    

    #contructing model from layers

    model = Model(inputs=input_layer,

                  outputs=output1)

    

    #compiling model

    model.compile(loss='binary_crossentropy',

                  optimizer='adam',

                  metrics=['accuracy']

                  )

    

    return model
#creating instance of model

model = create_model(input_shape=INPUT_DIM,

                     output_shape=OUTPUT_DIM)
model.summary()
#initializing generator for training

generator = generate(image_directory=image_dir,

                     labels=y,

                     train_idx=train_idx,

                     batch_size=BATCH_SIZE, 

                     size=IMAGE_SIZE)
#fitting model

history = model.fit(generator, 

                    epochs=EPOCHS, 

                    steps_per_epoch=STEPS)
def predict(image_directory, indices=None, plot=False):

    

    """

    Function to predict all images in a given path

    

    Parameters

    ----------

    image_directory: str

        Path for images to be predicted

    indices: list, default = None

        Indices corresponding to image labels to predict

    plot: boolean, default=False

        Whether or not to create plot of predictions

        

    Returns

    -------

    predictions_list: list

        List of predictions corresponding to the images

    """

    

    images_list = []

    

    prebuilt_model = xception.Xception(include_top=True,

                                       weights='imagenet')           #loading pre-built model

    

    xception_model = Model(inputs=prebuilt_model.input,

                           outputs=prebuilt_model.layers[-2].output) #repurposing pre-built model for feature extraction

    

    if indices!=None:

        image_filenames = [str(idx) + '.jpg' for idx in indices]

    else:

        image_filenames = os.listdir(image_directory)

    

    if plot:

        dim = int(np.ceil(np.sqrt(len(image_filenames))))

        fig, axs = plt.subplots(nrows=dim, 

                                ncols=dim,

                                figsize=(20, 14))

        plt.suptitle('Example of Model Predictions', fontsize=32)

        

#         axs = axs.flatten()

        

    

    #looping over all images in path

    for i, file in enumerate(image_filenames):

        



        img = image.load_img(os.path.join(image_directory,

                                          file))                     #loading images

        img_arr = image.img_to_array(img, dtype='float32')           #converting images to array

    

        if plot:

            axs.flat[i].imshow(img_arr/255.0)

            

        images_list.append(img_arr)

        gc.collect()       

    

    print('preprocessing...')

    images_list = np.array(images_list)

    img_arr = xception.preprocess_input(images_list)                 #preprocessing image array using xception method

    print('extracting features...')

    img_features = xception_model.predict(img_arr)                   #extracting features from image using prebuilt xception model

    img_features = np.array(img_features)

    print('predicting...')

    prediction = model.predict(img_features)                         #predicting time from image features                        

    hour_max = np.argmax(prediction[:, :12], axis=1)                 #obtaining hour with the highest probability

    minute_max = np.argmax(prediction[:, 12:], axis=1) + 12          #obtaining minute with the highest probability

    prediction_list = [(binarizer.classes_[x],                       #getting labels for predictions for binarizer

                        binarizer.classes_[y]) 

                        for (x,y) in list(zip(hour_max, minute_max))]



    if plot:                                                         #setting title for plots

        for i, v in enumerate(prediction_list):

            axs.flat[i].set_title(str(v[0]) + ' ' + str(v[1]))

            axs.flat[i].axis('off')

        for j in range(i+1, dim**2):                                 #removing excess subplots

              fig.delaxes(axs.flat[j])

            

    return prediction_list

#predicting samples used in earlier vizualization

predictions = predict('/kaggle/input/analog-clocks/analog_clocks/samples/',

                        plot=True)
#taking sample of test set to visualize results on unseen data

#this is done due to memory limitations

SIZE = 64

sample_test = list(np.random.choice(test_idx, size=SIZE))
#predicting sample of test set

predictions = predict('/kaggle/input/analog-clocks/analog_clocks/images/',

                        indices=sample_test,

                        plot=True)
sample_results = pd.DataFrame(list(zip(labels_df.loc[sample_test]['tuples'].values, pd.Series(predictions))), columns=['Actual', 'Predicted'])

sample_results
#saving model

for layer in model.layers:

    layer.trainable = False

model.save('model')
import pandas as pd

import cv2, os              

import numpy as np         

import matplotlib.pyplot as plt

from random import shuffle 

from tqdm import tqdm      
TRAIN_DIR = '../input/train/'

TEST_DIR = '../input/test/'



train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]

train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]
train_images = train_dogs + train_cats

shuffle(train_images) # mixing up the training data
train_images[:5] # check out the format of the raw training images labels
def labeling(img):

    image_label = img.split('.')[-3].split('/')[-1] # split the raw images labels to only "dog" or "cat", then convert to an array like below:

    if image_label == 'cat': return [1,0] 

    elif image_label == 'dog': return [0,1]
# process the training images and their labels into arrays:

from keras import preprocessing

width = 64

height = 64

def train_data_generate():

    training_data = []

    for image in tqdm(train_images):

        label = labeling(image)

        img = preprocessing.image.load_img(image, target_size=(width, height))

        x = preprocessing.image.img_to_array(img)

        training_data.append([np.array(x), np.array(label)])

    return training_data
# process the testing images and their labels into arrays:

from glob import glob

def test_data_generate():

    testing_data = []

    for image in glob('../input/test/*.*'):

        number = image.split('.')[2].split('/')[-1]

        image = preprocessing.image.load_img(image, target_size=(width, height))

        x = preprocessing.image.img_to_array(image)

        testing_data.append([np.array(x), number])

        shuffle(testing_data)

    return testing_data
train_data = train_data_generate()
# define the training data and target

X = np.array([i[0] for i in train_data])/255.0

Y = np.array([i[1] for i in train_data])
#model with 2 convolution layers

from keras.models import Sequential

from keras.layers.core import Activation, Dropout, Flatten, Dense

from keras.layers.convolutional import Convolution2D, MaxPooling2D

from keras.optimizers import Adam



conv1 = 32

conv1_drop = 0.2

conv2 = 64

conv2_drop = 0.2

dense1_n = 1024

dense1_drop = 0.2

dense2_n = 512

dense2_drop = 0.2

lr = 0.001



epochs = 20

batch_size = 32

color_channels = 3



def build_model_1(conv_1_drop=conv1_drop, conv_2_drop=conv2_drop, dense1_n=dense1_n, dense1_drop=dense1_drop, dense2_n=dense2_n, dense2_drop=dense2_drop):

    model_1 = Sequential()

    

    model_1.add(Convolution2D(conv1, (3, 3), input_shape=(width, height, color_channels), activation='relu'))

    model_1.add(MaxPooling2D(pool_size=(2, 2)))

    model_1.add(Dropout(conv_1_drop))

    

    model_1.add(Convolution2D(conv2, (3, 3), activation='relu'))

    model_1.add(MaxPooling2D(pool_size=(2, 2)))

    model_1.add(Dropout(conv_2_drop))

    

    model_1.add(Flatten())

    

    model_1.add(Dense(dense1_n, activation='relu'))

    model_1.add(Dropout(dense1_drop))

    

    model_1.add(Dense(dense2_n, activation='relu'))

    model_1.add(Dropout(dense2_drop))

    

    model_1.add(Dense(2, activation='softmax'))

    

    model_1.compile(loss='binary_crossentropy',

                  optimizer=Adam(lr=lr),

                  metrics=['accuracy'])

    return model_1

#model with 3 convolution layers

conv3 = 128

conv3_drop = 0.2



def build_model_2(conv_1_drop=conv1_drop, conv_2_drop=conv2_drop, conv_3_drop=conv3_drop, 

                dense1_n=dense1_n, dense1_drop=dense1_drop, dense2_n=dense2_n, dense2_drop=dense2_drop):

    model_2 = Sequential()

    

    model_2.add(Convolution2D(conv1, (3, 3), input_shape=(width, height, color_channels), activation='relu'))

    model_2.add(MaxPooling2D(pool_size=(2, 2)))

    model_2.add(Dropout(conv_1_drop))

    

    model_2.add(Convolution2D(conv2, (3, 3), activation='relu'))

    model_2.add(MaxPooling2D(pool_size=(2, 2)))

    model_2.add(Dropout(conv_2_drop))

    

    model_2.add(Convolution2D(conv3, (3, 3), activation='relu'))

    model_2.add(MaxPooling2D(pool_size=(2, 2)))

    model_2.add(Dropout(conv_3_drop))

    

    model_2.add(Flatten())

    

    model_2.add(Dense(dense1_n, activation='relu'))

    model_2.add(Dropout(dense1_drop))

    

    model_2.add(Dense(dense2_n, activation='relu'))

    model_2.add(Dropout(dense2_drop))

    

    model_2.add(Dense(2, activation='softmax'))

    

    model_2.compile(loss='binary_crossentropy',

                 optimizer=Adam(lr=lr),

                  metrics=['accuracy'])

    return model_2

#model with 4 convolution layers

conv4 = 256

conv4_drop = 0.2



def build_model_3(conv_1_drop=conv1_drop, conv_2_drop=conv2_drop, conv_3_drop=conv3_drop, conv_4_drop=conv4_drop, 

                dense1_n=dense1_n, dense1_drop=dense1_drop, dense2_n=dense2_n, dense2_drop=dense2_drop):

    model_3 = Sequential()

    

    model_3.add(Convolution2D(conv1, (3, 3), input_shape=(width, height, color_channels), activation='relu'))

    model_3.add(MaxPooling2D(pool_size=(2, 2)))

    model_3.add(Dropout(conv_1_drop))

    

    model_3.add(Convolution2D(conv2, (3, 3), activation='relu'))

    model_3.add(MaxPooling2D(pool_size=(2, 2)))

    model_3.add(Dropout(conv_2_drop))

    

    model_3.add(Convolution2D(conv3, (3, 3), activation='relu'))

    model_3.add(MaxPooling2D(pool_size=(2, 2)))

    model_3.add(Dropout(conv_3_drop))

    

    model_3.add(Convolution2D(conv4, (3, 3), activation='relu'))

    model_3.add(MaxPooling2D(pool_size=(2, 2)))

    model_3.add(Dropout(conv_4_drop))

    

    model_3.add(Flatten())

    

    model_3.add(Dense(dense1_n, activation='relu'))

    model_3.add(Dropout(dense1_drop))

    

    model_3.add(Dense(dense2_n, activation='relu'))

    model_3.add(Dropout(dense2_drop))

    

    model_3.add(Dense(2, activation='softmax'))

    

    model_3.compile(loss='binary_crossentropy',

                 optimizer=Adam(lr=lr),

                  metrics=['accuracy'])

    return model_3

model_one = build_model_1()

model_one.summary()
model_two = build_model_2()

model_two.summary()
model_three = build_model_3()

model_three.summary()
model_one.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
model_two.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
model_three.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
test_data = test_data_generate()
test_img = np.array([i[0] for i in test_data])/255.0

test_id = np.array([i[1] for i in test_data])

test_img.shape
predictions_1 = model_one.predict(test_img, verbose=1)
for i in range(0, 10):

    if predictions_1[i, 1] >= 0.5: 

        print('I am {:.2%} sure this is a Dog'.format(predictions_1[i][1]))

    else: 

        print('I am {:.2%} sure this is a Cat'.format(1-predictions_1[i][1]))

        

    plt.imshow(test_img[i])

    plt.show()
results = pd.DataFrame({

    'id': test_id,

    'label': predictions_1[:, 1]

})



results.to_csv('sample_submission_1.csv',index=False)

sub = pd.read_csv('sample_submission_1.csv')

sub.head(5)
predictions_2 = model_two.predict(test_img, verbose=1)
for i in range(0, 10):

    if predictions_2[i, 1] >= 0.5: 

        print('I am {:.2%} sure this is a Dog'.format(predictions_2[i][1]))

    else: 

        print('I am {:.2%} sure this is a Cat'.format(1-predictions_2[i][1]))

        

    plt.imshow(test_img[i])

    plt.show()
results = pd.DataFrame({

    'id': test_id,

    'label': predictions_2[:, 1]

})



results.to_csv('sample_submission_2.csv',index=False)

sub = pd.read_csv('sample_submission_2.csv')

sub.head(5)
predictions_3 = model_three.predict(test_img, verbose=1)
for i in range(0, 10):

    if predictions_3[i, 1] >= 0.5: 

        print('I am {:.2%} sure this is a Dog'.format(predictions_3[i][1]))

    else: 

        print('I am {:.2%} sure this is a Cat'.format(1-predictions_3[i][1]))

        

    plt.imshow(test_img[i])

    plt.show()
results = pd.DataFrame({

    'id': test_id,

    'label': predictions_3[:, 1]

})



results.to_csv('sample_submission_3.csv',index=False)

sub = pd.read_csv('sample_submission_3.csv')

sub.head(5)
import os

import numpy as np

import pandas as pd

import cv2

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



from sklearn.model_selection import train_test_split

import itertools



from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.models import Sequential, Model

from keras.layers import Dense, Activation, Flatten, Dropout, concatenate, Input, Conv2D, MaxPooling2D

from keras.optimizers import Adam, Adadelta

from keras.layers.advanced_activations import LeakyReLU

from keras.utils.np_utils import to_categorical
train_dir = '../input/plant-seedlings-classification/train'

test_dir = '../input/plant-seedlings-classification/test'

sample_submission = pd.read_csv('../input/plant-seedlings-classification/sample_submission.csv')

SPECIES = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen',

              'Loose Silky-bent', 'Maize', 'Scentless Mayweed', 'Shepherds Purse',

              'Small-flowered Cranesbill', 'Sugar beet']



for species in SPECIES:

    print('{} {} images'.format(species, len(os.listdir(os.path.join(train_dir, species)))))

    
train = []



for species_num, species in enumerate(SPECIES):

    for file in os.listdir(os.path.join(train_dir, species)):

        train.append(['train/{}/{}'.format(species, file), species_num, species])

        

train = pd.DataFrame(train, columns=['file', 'species_num', 'species'])



print('Training Data: ',train.shape)

def create_mask_for_plant(image):

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)



    sensitivity = 35

    lower_hsv = np.array([60 - sensitivity, 100, 50])

    upper_hsv = np.array([60 + sensitivity, 255, 255])



    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    

    return mask



def segment_plant(image):

    mask = create_mask_for_plant(image)

    output = cv2.bitwise_and(image, image, mask = mask)

    return output



def sharpen_image(image):

    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)

    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)

    return image_sharp

%%time



x_train = []



for species in SPECIES:

    lis =  os.listdir(os.path.join(train_dir, species))

    new = []

    

    for l in lis:

        new.append(os.path.join(train_dir, species) + '/' + l)

        

    for l in new:

        img = cv2.imread(l)

        img = cv2.resize(img,dsize=(256,256))

        img_stack = segment_plant(img)

        img_stack = sharpen_image(img_stack)

        img_stack = cv2.cvtColor( img_stack, cv2.COLOR_RGB2GRAY )

        img_stack = np.reshape(img_stack,(256,256,1))

        x_train.append(np.concatenate((np.array(img),np.array(img_stack)),axis=2))

        



x_train = np.array(x_train)

labels = train['species_num']

labels = to_categorical(labels, num_classes = len(SPECIES))

x_train, x_val, y_train, y_val = train_test_split(x_train, labels, test_size = 0.1, random_state=10)

input_shape = x_train[1].shape

print('Input Shape is :', input_shape)

def fire_incept(x, fire=16, intercept=64):

    x = Conv2D(fire, (5,5), strides=(2,2))(x)

    x = LeakyReLU(alpha=0.15)(x)

    

    left = Conv2D(intercept, (3,3), padding='same')(x)

    left = LeakyReLU(alpha=0.15)(left)

    

    right = Conv2D(intercept, (5,5), padding='same')(x)

    right = LeakyReLU(alpha=0.15)(right)

    

    x = concatenate([left, right], axis=3)

    return x



def fire_squeeze(x, fire=16, intercept=64):

    x = Conv2D(fire, (1,1))(x)

    x = LeakyReLU(alpha=0.15)(x)

    

    left = Conv2D(intercept, (1,1))(x)

    left = LeakyReLU(alpha=0.15)(left)

    

    right = Conv2D(intercept, (3,3), padding='same')(x)

    right = LeakyReLU(alpha=0.15)(right)

    

    x = concatenate([left, right], axis=3)

    return x



image_input=Input(shape=input_shape)



x = fire_incept((image_input), fire=16, intercept=16)



x = fire_incept(x, fire=32, intercept=32)

x = fire_squeeze(x, fire=32, intercept=32)



x = fire_incept(x, fire=64, intercept=64)

x = fire_squeeze(x, fire=64, intercept=64)



x = fire_incept(x, fire=64, intercept=64)

x = fire_squeeze(x, fire=64, intercept=64)



x = Conv2D(64, (3,3))(x)

x = LeakyReLU(alpha=0.1)(x)



x = Flatten()(x)



x = Dense(512)(x)

x = LeakyReLU(alpha=0.1)(x)

x = Dropout(0.1)(x)



out = Dense(len(SPECIES), activation='softmax')(x)



model_new = Model(image_input, out)

model_new.summary()

model_new.compile(optimizer = Adam(lr=.00025) , loss = 'categorical_crossentropy', metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, 

                                            factor=0.5, min_lr=0.00001)

datagen = ImageDataGenerator(rotation_range=40, zoom_range = 0.2, width_shift_range=0.2, height_shift_range=0.2,

                             horizontal_flip=True, vertical_flip=True)

datagen.fit(x_train)

model_new.load_weights('../input/pretrained-weight/model_weights.h5f')

batch_size = 32

epochs = 40

history = model_new.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size), epochs = epochs,

                                  validation_data = (x_val,y_val), verbose = 1, 

                                  steps_per_epoch=x_train.shape[0] // batch_size, 

                                  callbacks=[learning_rate_reduction])
model_new.save_weights('./model_weights_2.h5f', overwrite=True)
%%time



test = []

for file in os.listdir(os.path.join(test_dir)):

    test.append(test_dir + '/' + file)



for i in range(len(test)):

    img = cv2.imread(test[i])

    img = cv2.resize(img,dsize=(256,256))

    img_stack = segment_plant(img)

    img_stack = sharpen_image(img_stack)

    img_stack = cv2.cvtColor( img_stack, cv2.COLOR_RGB2GRAY )

    img_stack = np.reshape(img_stack,(256,256,1))

    x_test.append(np.concatenate((np.array(img),np.array(img_stack)),axis=2))



x_test = np.array(x_test)

Pred_labels = np.argmax(model_new.predict(x_test),axis = 1)

Pred_labels = pd.DataFrame(Pred_labels,index =None,columns=['species_num'])



test_id = []

for file in os.listdir(test_dir):

    test_id.append(['{}'.format(file)])



test_id = pd.DataFrame(test_id, columns=['file'])



test_df = pd.DataFrame()

test_df['species_num'] = Pred_labels['species_num']

test_df['file'] = test_id['file']

test_df['species'] = [SPECIES[i] for i in Pred_labels['species_num']]



submission = pd.merge(left=sample_submission, right=test_df[['file', 'species']], on="file", how="right")

submission.drop(['species_x'], axis = 1, inplace = True)

submission.columns = ['file','species'] 



submission.to_csv('./submission.csv', index=False)

print(submission.head())
